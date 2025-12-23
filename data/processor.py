VOCAB = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<EOS>", "<PAD>"]

stoi = {s: i for i, s in enumerate(VOCAB)}
itos = {i: s for i, s in enumerate(VOCAB)}

PAD_ID = stoi["<PAD>"]
EOS_ID = stoi["<EOS>"]
