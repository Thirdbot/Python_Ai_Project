text= "Hi there, buddy! ?"
text2 = "Hey! How's it going?"

slide_token = 1

def slide_window(inputs,slide):
    for i in range(0,len(inputs),slide):
        yield inputs[0:i+slide]

texts = slide_window(text,slide_token)
texts2 = slide_window(text2,slide_token)

def visual_model(inputs,src):
    if inputs in src:
        word = src[src.index(inputs[-1]):]
        print(word)
        source = word[1%len(word)]
        word = word[src.index(source)+1:]
        print(word)

        return source
for i in texts:
    output = visual_model(i,text)
    print(f"{i} -> {output}")
