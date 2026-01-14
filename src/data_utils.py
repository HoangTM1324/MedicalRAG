def format_data_for_training(example, tokenizer):
    """Hàm map dataset cho TrainingWithQLoRA."""
    label2id = {"yes": 0, "no": 1, "maybe": 2}
    
    contexts = example.get('context', {})
    full_context = "\n".join(contexts['contexts']) if 'contexts' in contexts else ""
    question = example.get('question', '')
    
    text = f"Context:\n{full_context}\n\nQuestion:\n{question}\n\nAnswer:"
    
    tokenized = tokenizer(text, truncation=True, max_length=512, padding=False)
    
    decision = example.get('final_decision', 'maybe')
    tokenized["labels"] = label2id.get(decision.lower(), 2)
    
    return tokenized
