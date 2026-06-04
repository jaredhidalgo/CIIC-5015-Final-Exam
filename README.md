# CIIC 5015 Final Exam


Introduction to AI: Final Assignment Starter Code
Questions 1 and 2,: Attention, Transformers & LLMs



FILES IN THIS PACKAGE
---------------------

  models.py        Q1  — Implement AttentionBlock
  gpt_model.py     Q2  — Implement Transformer_Block.forward
                          and GPT.forward
  train_gpt.py     Q1  — Training script (run after implementing Q7)
  q2_challenge.py  Q2Ch  — Starter code for all three challenge options
  README.txt       This file

DEPENDENCIES
------------

  pip install torch numpy matplotlib

  If you encounter NumPy errors:
  pip install numpy==1.24.3

QUICK START
-----------

Step 1 — Implement Q1
  Edit models.py
  Fill in AttentionBlock.__init__ and AttentionBlock.forward
  Test your implementation (shape check only):
      python models.py

Step 2 — Implement Q2
  Edit gpt_model.py
  Fill in Transformer_Block.forward  (Part A)
  Fill in GPT.forward                (Part B)
  Test shapes:
      python gpt_model.py

Step 3 — Train your model (Q1)
  Get a text corpus (min 500,000 characters) and save as input.txt
  Corpus suggestions:
      Shakespeare : https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
      Wikipedia   : https://dumps.wikimedia.org (extract plain text)
      News/lyrics : Kaggle datasets
  Run:
      python train_gpt.py


Step 5 — Written analysis
 Answer questions and provide analysis
HYPERPARAMETERS (edit in train_gpt.py)
---------------------------------------

  BLOCK_SIZE    = 64      context window in characters
  LAYER_SIZE    = 64      embedding / hidden dimension
  N_LAYERS      = 2       number of Transformer blocks
  LEARNING_RATE = 3e-4    Adam optimizer learning rate
  MAX_ITERS     = 3000    total training steps
  BATCH_SIZE    = 32      sequences per batch

Start with these defaults. Once training works, experiment with
larger values and observe the effect on generated text quality.

SHAPE REFERENCE
---------------

  After AttentionBlock:
      input:  (batch_size, seq_len, layer_size)
      output: (batch_size, seq_len, layer_size)

  After Transformer_Block:
      input:  (batch_size, seq_len, layer_size)
      output: (batch_size, seq_len, layer_size)   [same shape]

  After GPT.forward:
      input:  (batch_size, seq_len)               [integer indices]
      output: (batch_size, seq_len, vocab_size)   [float logits]

SUBMISSION
----------

Upload to the course portal:
  models.py         your Q6 implementation
  gpt_model.py      your Q7 implementation
 written analysis and answers

One submission per team. Tag your partner on the course portal.

TIPS
----

  - Run python models.py and python gpt_model.py before training
    to catch shape errors early without waiting for training to fail.
  - If training loss goes to NaN or Inf, your learning rate is too
    high or there is a bug in your attention scaling.
  - Both partners must be able to explain every line of code at
    the check-in session — divide work so both understand all of it.


