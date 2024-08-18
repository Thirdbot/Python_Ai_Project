test_data = [None,"ad",4,[]]
temp = test_data

test_data =temp
test_data = [str(v) if not isinstance(v, str) else v for v in test_data]
print(test_data)