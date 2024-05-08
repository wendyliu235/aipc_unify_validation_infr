import keras
import keras_nlp
import time
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-p', '--precision',
                   type=str, help='model_precision')
args = parser.parse_args()

keras.config.set_floatx(args.precision)
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("code_gemma_2b_en")
gemma_lm.summary()



BEFORE_CURSOR = "<|fim_prefix|>"
AFTER_CURSOR = "<|fim_suffix|>"
AT_CURSOR = "<|fim_middle|>"
FILE_SEPARATOR = "<|file_separator|>"
END_TOKEN = gemma_lm.preprocessor.tokenizer.end_token

stop_tokens = (BEFORE_CURSOR, AFTER_CURSOR, AT_CURSOR, FILE_SEPARATOR, END_TOKEN)

stop_token_ids = tuple(gemma_lm.preprocessor.tokenizer.token_to_id(x) for x in stop_tokens)


def format_completion_prompt(before, after):
    return f"{BEFORE_CURSOR}{before}{AFTER_CURSOR}{after}{AT_CURSOR}"

before = "import "
after = """if __name__ == "__main__":\n    sys.exit(0)"""
prompt = format_completion_prompt(before, after)
print(prompt)

total_time = 0.0
num_iter = 10
num_warmup = 3
total_list = []
for i in range(num_iter):
  tic = time.time()
  output = gemma_lm.generate(
    prompt, stop_token_ids=stop_token_ids, max_length=128
  )
  toc = time.time()
  print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
  if i >= num_warmup:
    total_time += toc - tic

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
