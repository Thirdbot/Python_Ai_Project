# def gen1():
#     LL=[]
#     for i in range(3):
#         L = []
#         for i in range(10):
#             L.append(i)
#         LL.append(L)
#     yield LL

# batch = 2
# def gen2(data):
#     for i in data:
#         print("Orig:",i)
#         for idx in range(0,len(i),batch):
#             batched = i[idx:min(idx+batch,len(i))]
#             yield batched


# saved = gen2(gen1())

# for i in saved:
#     print("New batched: ",i)

sent = input("Enter: ")

sent = list(sent.split())
print(sent)