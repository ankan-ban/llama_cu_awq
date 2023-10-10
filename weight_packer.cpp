#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

constexpr int group_size = 128;    // hardcoded as this implementation only supports 128

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    float rope_theta; // theta for the rope rotational embedding
} Config;


// extract config from config.json file
void getConfig(Config* pConfig, char* json)
{
    char* p = strstr(json, "\"hidden_size\":");
    if (!p) { printf("error parsing config.json hidden_size not found"); exit(1); }
    p += strlen("\"hidden_size\":");
    pConfig->dim = atoi(p);

    p = strstr(json, "\"intermediate_size\":");
    if (!p) { printf("error parsing config.json intermediate_size not found"); exit(1); }
    p += strlen("\"intermediate_size\":");
    pConfig->hidden_dim = atoi(p);

    p = strstr(json, "\"num_hidden_layers\":");
    if (!p) { printf("error parsing config.json num_hidden_layers not found"); exit(1); }
    p += strlen("\"num_hidden_layers\":");
    pConfig->n_layers = atoi(p);

    p = strstr(json, "\"num_attention_heads\":");
    if (!p) { printf("error parsing config.json num_attention_heads not found"); exit(1); }
    p += strlen("\"num_attention_heads\":");
    pConfig->n_heads = atoi(p);

    p = strstr(json, "\"num_key_value_heads\":");
    if (!p) {
        pConfig->n_kv_heads = pConfig->n_heads;
    } else {
        p += strlen("\"num_key_value_heads\":");
        pConfig->n_kv_heads = atoi(p);
    }

    p = strstr(json, "\"vocab_size\":");
    if (!p) { printf("error parsing config.json vocab_size not found"); exit(1); }
    p += strlen("\"vocab_size\":");
    pConfig->vocab_size = atoi(p);

    p = strstr(json, "\"max_position_embeddings\":");
    if (!p) { printf("error parsing config.json max_position_embeddings not found"); exit(1); }
    p += strlen("\"max_position_embeddings\":");
    pConfig->seq_len = atoi(p);

    p = strstr(json, "\"rope_theta\":");
    if (!p) {
        pConfig->rope_theta = 10000.0f;
    } else {
        p += strlen("\"rope_theta\":");
        pConfig->rope_theta = atof(p);
    }

    printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\nrope_theta: %g\n",
        pConfig->dim, pConfig->hidden_dim, pConfig->n_heads, pConfig->n_kv_heads, pConfig->n_layers, pConfig->seq_len, pConfig->vocab_size, pConfig->rope_theta);
}

void getFileContents(void* buf, char* filename, size_t bytes) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { printf("\nUnable to open %s\n", filename); exit(1); }
    if (fread(buf, 1, bytes, fp) != bytes) { printf("error reading weights from %s", filename);  exit(1); }
    fclose(fp);
}

void copyInputFileToFile(FILE* fp, char* filename, size_t bytes) {
    void* buf = malloc(bytes);
    getFileContents(buf, filename, bytes);
    if(fwrite(buf, 1, bytes, fp) != bytes) { printf("error writing output file from input %s", filename);  exit(1); }
    free(buf);
}

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

// 1. Convert from row-major to col-major
// 2. Get rid of the order_map (simply pack as little endian)
void repack_q_data(uint32_t* q_weight_out, const uint32_t* q_weight_in, int height, int width) {
    uint32_t* temp = (uint32_t*)malloc(width * height * sizeof(uint32_t));
    int order_map[] = { 0, 2, 4, 6, 1, 3, 5, 7 };   // used by AWQ's original implementation

    // 1. convert to uint32 col-major array first (only 4 LSBs of each element are non-zero)
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x += 8) {
            uint32_t packed_q_wt = q_weight_in[(y * width + x) / 8];
            uint32_t q[8];
            for (int i = 0; i < 8; i++)
            {
                uint32_t q_wt = packed_q_wt & 0xF;
                q[order_map[i]] = q_wt;
                packed_q_wt = packed_q_wt >> 4;
            }

            for (int i = 0; i < 8; i++)
                temp[(x + i) * height + y] = q[i];  // note - transpose here
        }

    // 2. pack 8 consecutive elements to single uint32_t (consecutive in the inner-most dimension which is the column dimension)
    int packed_wt_height = divUp(height, 8);
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y += 8) {
            uint32_t packed_val = 0;
            for (int i = 0; i < 8; i++) {
                packed_val = (packed_val) | (temp[x * height + y + i] << (4 * i));
            }
            int packed_wt_y = y / 8;
            q_weight_out[x * packed_wt_height + packed_wt_y] = packed_val;
        }

    free(temp);
}

void repack_q_weights(uint32_t* q_weight_out, uint32_t* q_zeros_out, uint16_t* scales_out,
    const uint32_t* q_weight_in, const uint32_t* q_zeros_in, const uint16_t* scales_in, int height, int width, int group_size)
{
    // weights
    repack_q_data(q_weight_out, q_weight_in, height, width);

    int meta_height = divUp(height, group_size);

    // zeros
    repack_q_data(q_zeros_out, q_zeros_in, meta_height, width);

    // scales
    for (int x = 0; x < width; x++)
        for (int y = 0; y < meta_height; y++)
            scales_out[x * meta_height + y] = scales_in[y * width + x];
}

void repackQWeightByName(FILE *fp, char* fileNameBase, const char* weightName, size_t height, size_t width, bool needsRepacking) {
    uint32_t* qweight;
    uint32_t* qzeros;
    uint16_t* scales;

    size_t orig_qweight_bytes = 0;
    size_t orig_qzeros_bytes = 0;
    size_t orig_scales_bytes = 0;

    int meta_height = divUp(height, group_size);
    int packed_wt_height = divUp(height, 8);
    int packed_zeros_height = divUp(meta_height, 8);

    if (needsRepacking) {
        size_t orig_qw_width = divUp(width, 8);                 // eight 4-bit elements packed in single uint32_t
        size_t orig_meta_height = divUp(height, group_size);    // group-count

        orig_qweight_bytes = orig_qw_width * height * sizeof(uint32_t);
        orig_qzeros_bytes = orig_qw_width * orig_meta_height * sizeof(uint32_t);
        orig_scales_bytes = orig_meta_height * width * sizeof(uint16_t);
    } else {
        int orig_scales_height = packed_zeros_height * 8;        // AWQ repo adds extra padding to scales to make it a multiple of 8
        orig_qweight_bytes = packed_wt_height * width * sizeof(uint32_t);
        orig_qzeros_bytes = packed_zeros_height * width * sizeof(uint32_t);
        orig_scales_bytes = orig_scales_height * width * sizeof(uint16_t);
    }

    qweight = (uint32_t*)malloc(orig_qweight_bytes);
    qzeros = (uint32_t*)malloc(orig_qzeros_bytes);
    scales = (uint16_t*)malloc(orig_scales_bytes);

    char filename[512];
    sprintf(filename, "%s.%s.qweight.bin", fileNameBase, weightName);
    getFileContents(qweight, filename, orig_qweight_bytes);
    sprintf(filename, "%s.%s.qzeros.bin", fileNameBase, weightName);
    getFileContents(qzeros, filename, orig_qzeros_bytes);
    sprintf(filename, "%s.%s.scales.bin", fileNameBase, weightName);
    getFileContents(scales, filename, orig_scales_bytes);

    // first convert to weights that are easy to de-quantize with our simple kernel
    if (needsRepacking) {
        uint32_t* q_weight_t = (uint32_t*)malloc(packed_wt_height * width * sizeof(uint32_t));
        uint32_t* q_zeros_t = (uint32_t*)malloc(packed_zeros_height * width * sizeof(uint32_t));
        uint16_t* scales_t = (uint16_t*)malloc(meta_height * width * sizeof(uint16_t));

        repack_q_weights(q_weight_t, q_zeros_t, scales_t, qweight, qzeros, scales, height, width, group_size);

        free(qweight);
        free(qzeros);
        free(scales);

        qweight = q_weight_t;
        qzeros = q_zeros_t;
        scales = scales_t;
    }
    else {
        // AWQ repo adds extra padding to scales to make it a multiple of 8
        // We need to get rid of that padding
        int orig_scales_height = packed_zeros_height * 8;
        uint16_t* scales_t = (uint16_t*)malloc(meta_height * width * sizeof(uint16_t));
        for (int x = 0; x < width; x++)
            for (int y = 0; y < meta_height; y++)
                scales_t[x * meta_height + y] = scales[x * orig_scales_height + y];

        free(scales);
        scales = scales_t;
    }

    // dump to file
    if (fwrite(qweight, 1, packed_wt_height * width * sizeof(uint32_t), fp) != packed_wt_height * width * sizeof(uint32_t))  { 
        printf("error writing output q_weights file from input %s", fileNameBase);  exit(1); 
    }
    if (fwrite(qzeros, 1, packed_zeros_height * width * sizeof(uint32_t), fp) != packed_zeros_height * width * sizeof(uint32_t)) {
        printf("error writing output q_zeros file from input %s", fileNameBase);  exit(1);
    }
    if (fwrite(scales, 1, meta_height * width * sizeof(uint16_t), fp) != meta_height * width * sizeof(uint16_t)) {
        printf("error writing output scales file from input %s", fileNameBase);  exit(1);
    }

    if (needsRepacking) {
        free(qweight);
        free(qzeros);
    }
    free(scales);
}

char config_json[1024 * 1024];
int main(int argc, char *argv[])
{
    if (argc != 5) { printf("usage: weight_packer <config.json from huggingface> <path_to_awq_bin_weights> <output_bin_filename> [OldAwqFormat: 0 or 1]\n"); return 0; }

    char *config_file_name = argv[1];
    char* input_dir = argv[2];
    char* op_file_name = argv[3];
    int old_awq_format = atoi(argv[4]);     // old AWQ format needs weight repacking

    // read the config file
    Config config;
    FILE* fp_config;
    fp_config = fopen(config_file_name, "rb");
    if (!fp_config) { printf("unable to open config file\n"); return 0; }
    if(fread(config_json, 1, sizeof(config_json), fp_config) == 0) { printf("unable to read config file\n"); return 0; }
    fclose(fp_config);
    getConfig(&config, config_json);

    FILE* fp;
    fp = fopen(op_file_name, "wb+");
    if (!fp) { printf("unable to open output file\n"); return 0; }

    // write the header
    if (fwrite(&config, sizeof(config), 1, fp) != 1) { printf("unable to write model metadata\n"); return 0; }

    char fileNameBase[512];
    char filename[512];

    sprintf(filename, "%s/model.embed_tokens.weight.bin", input_dir);
    copyInputFileToFile(fp, filename, config.vocab_size * config.dim * sizeof(uint16_t));

    sprintf(filename, "%s/lm_head.weight.bin", input_dir);
    copyInputFileToFile(fp, filename, config.vocab_size * config.dim * sizeof(uint16_t));

    sprintf(filename, "%s/model.norm.weight.bin", input_dir);
    copyInputFileToFile(fp, filename, config.dim * sizeof(uint16_t));

    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

    for (int i = 0; i < config.n_layers; i++)
    {
        printf("\nProcessing weights for layer: %d\n", i);

        sprintf(fileNameBase, "%s/model.layers.%d", input_dir, i);

        repackQWeightByName(fp, fileNameBase, "self_attn.q_proj", config.dim, config.dim, old_awq_format);
        repackQWeightByName(fp, fileNameBase, "self_attn.k_proj", config.dim, kv_dim, old_awq_format);
        repackQWeightByName(fp, fileNameBase, "self_attn.v_proj", config.dim, kv_dim, old_awq_format);
        repackQWeightByName(fp, fileNameBase, "self_attn.o_proj", config.dim, config.dim, old_awq_format);

        repackQWeightByName(fp, fileNameBase, "mlp.up_proj", config.dim, config.hidden_dim, old_awq_format);
        repackQWeightByName(fp, fileNameBase, "mlp.gate_proj", config.dim, config.hidden_dim, old_awq_format);
        repackQWeightByName(fp, fileNameBase, "mlp.down_proj", config.hidden_dim, config.dim, old_awq_format);

        sprintf(filename, "%s.input_layernorm.weight.bin", fileNameBase);
        copyInputFileToFile(fp, filename, config.dim * sizeof(uint16_t));

        sprintf(filename, "%s.post_attention_layernorm.weight.bin", fileNameBase);
        copyInputFileToFile(fp, filename, config.dim * sizeof(uint16_t));
    }

    printf("\nDone!\n");

    fclose(fp);
}