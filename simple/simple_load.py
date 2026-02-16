from transformers import AutoModelForCausalLM, AutoTokenizer

#1.加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
"/dmodels/Qwen2.5-Math-1.5B",
torch_dtype=torch.bfloat16,
attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-Math-1.5B")

#2.Forward pass
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)

logits = model(input_ids).logits
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

#3.save the model
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)

#4.Gradient accumulation（梯度累积）
#正常情况
for inputs, labels in data_loader:
    # Forward pass.
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    # Backward pass.
    loss.backward()
    # Update weights.
    optimizer.step()
    # Zero gradients in preparation for next iteration.
    optimizer.zero_grad()

#使用梯度累积，每4个batch更新一次权重，那么batch_size=16相当于batch_size=64的效果。
gradient_accumulation_steps = 4
for idx, (inputs, labels) in enumerate(data_loader):
    # Forward pass.
    logits = model(inputs)
    loss = loss_fn(logits, labels) / gradient_accumulation_steps
    # Backward pass.
    loss.backward()
    if (idx + 1) % gradient_accumulation_steps == 0:
    # Update weights every `gradient_accumulation_steps` batches.
    optimizer.step()
    # Zero gradients every `gradient_accumulation_steps` batches.
    optimizer.zero_grad()