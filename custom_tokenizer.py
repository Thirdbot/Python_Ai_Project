###this part is for custom tokenize load first
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from tokenizers.trainers import BpeTrainer

class Tokenization:
    
    def __init__(self) -> None:
        super().__init__()
        #load wikipedia datasets
        self.wiki_ds = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
        #save wiki ds for train
        with open("wiki.txt","w",encoding="utf-8") as f:
            for i in range(len(self.wiki_ds)):
                f.write(self.wiki_ds[i]["text"]+ "\n")
        #custom tokenizer
        #define unknown tokens and models to work with trainner like bpe,wordpiece etc...
        #tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer = Tokenizer(models.BPE())

        # tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        #      [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
        # )

        #normalise
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )

        #using Bert to seperate in space and punctuation
        #tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        #byte level seperation
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        #specials tokens defined
        special_tokens = ["[PAD]", "[CLS]","[SEP]"]
        #set trainers with special token seperately
        #trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
        
        #set trainers with one special tokens at the end
        trainer = trainers.BpeTrainer(vocab_size=24999, special_tokens=special_tokens)
        #get each batch sector from datasets to train tokenizer

        tokenizer.train_from_iterator(self.get_wiki_corpus(),trainer=trainer)

        #tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        #tokenizer.train(["wiki.txt"], trainer=trainer)


        ##define seperator between sentence
        # cls_token_id = tokenizer.token_to_id("[CLS]")
        # sep_token_id = tokenizer.token_to_id("[SEP]")

        # tokenizer.post_processor = processors.TemplateProcessing(
        #     single=f"[CLS]:0 $A:0 [SEP]:0",
        #     pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        #     special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
        # )
        cls_token_id = tokenizer.token_to_id("[CLS]")
        sep_token_id = tokenizer.token_to_id("[SEP]")
        pad_token_id = tokenizer.token_to_id("[PAD]")

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", cls_token_id), 
                ("[SEP]", sep_token_id),
                ("[PAD]", pad_token_id)
            ],
        )
        #include whitespace in merging
        # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)


        #set merge word
        #tokenizer.decoder = decoders.WordPiece(prefix="##")
        tokenizer.decoder = decoders.ByteLevel()
        #save tokenizer
        tokenizer.save("tokenizer.json")

    #batch datasets
    def get_wiki_corpus(self):
        for i in range(0, len(self.wiki_ds), 1000):
            yield self.wiki_ds[i : i + 1000]["text"]

if __name__ == "__main__":
    tokenization = Tokenization()