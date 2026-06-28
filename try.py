test_string = "你好 我爱你 你是谁!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(type(utf8_encoded))
print(type(test_string))
print(list(utf8_encoded))
print(len(test_string))
print(len(utf8_encoded))
answer = utf8_encoded.decode("utf-8") 
print(answer)     


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join([bytes([b]).decode("utf-8") for b in bytestring])



print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
print(decode_utf8_bytes_to_str_wrong(utf8_encoded))