# Viva preparation

**Which model produces game suggestions?** The saved CUAD word-count model. A bigram conditions on the preceding word and a trigram conditions on the preceding two. Both share the same fitted vocabulary and train on the same sentences.

**How are unseen phrases handled?** The model maps out-of-vocabulary words to a trainable unknown bucket. Each context distributes some probability according to the next lower order. With zero observations for that context, it uses the lower-order distribution entirely.

**How do you interpret perplexity?** It is the exponential of the average token negative log-likelihood. Lower is better for the same holdout, tokenization, and vocabulary. LexForge evaluates end-of-sentence tokens and reports unknown counts.

**What was actually fine-tuned?** GPT-2’s last transformer block, final normalization, and five-way head. The tokenizer and earlier model weights come from the pinned pretrained GPT-2 release. The serialized checkpoint stores the learned parameters and evaluation metadata.

**How did you control leakage?** CUAD sections follow their contract’s title-based split. LEDGAR keeps its published splits. Cross-split entries with equal normalized fingerprints are quarantined and the vocabulary comes only from the training split. Similar templates could still appear with different wording or titles.

**Why macro F1?** It gives each category equal weight despite the different category counts. Compare the confusion matrix and per-category precision and recall to see which categories account for errors.

**What happens with an unknown clause type?** The classifier always distributes its probability across the five supported labels, so an out-of-scope clause can receive a confident prediction. The application is educational and the score does not measure legal validity.

**What does a successful demo show?** Start with option 2 and a prefix such as “Either party”. Draft a sentence, compare the predicted and expected categories, and open My Progress. Restart with the same profile to show persistence. In Model Insights, distinguish token perplexity from classification metrics.
