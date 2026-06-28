import os
from collections import defaultdict,Counter
import regex as re # type: ignore
import json

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
  
  #1. 初始化基础词表
  # 词表从 0-255 的字节开始
  vocab = {i:bytes([i]) for i in range(256)}

  #计算需要进行的合并次数
  #词表大小 = 256 + 特殊token + 需要重新生成的Tokens
  num_merges = vocab_size - 256 - len(special_tokens)

  #2. 读取语料,并按特殊Token 分割
  with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

  # 如果制定了特殊token, 需要在开始统计之前将他们从语料中分割出来
  #防止BPE规则将特殊对的Token (如 <|endoftext|>)拆开或者跟普通文本混合。

  if special_tokens:
    #在正则中, | 表示 或, 将多个特殊token 连接起来,形成一个匹配任意token的正则
    special_regex = "|".join(re.escape(t) for t in special_tokens)

    #使用re.split 分割 
    parts = re.split(f"({special_regex})",text)

    #过滤从parts中提取出的特殊token本身,只保留用于BPE训练的部分
    train_segments = [p for p in parts if p not in special_tokens]
  else:
    train_segments = [text] #没有特殊token使用整个语料

  #3. 预分词(Pre-tokenization) 并统计词频
  #使用GPT-2的BPE预分词: 不允许跨类型合并,保护空格通常会把单词前面的空格和单词连在一起
  gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

  #raw_counts: 存储每个 "单词" (预分词后的结果) 以及出现频率
  #单词被表示为字节元组
  raw_counts = Counter()
  for segment in train_segments:
    #对每个单词使用预分词正则
    words = gpt2_pat.findall(segment)
    for word in words:
      #将一个单词转换成 UTF-8的字节序列
      raw_counts[tuple(bytes([b]) for b in word.encode("utf-8") )] += 1 #到这里统计完了词频

  words_list = []
  count_list = []
  for word_tuple, freq in raw_counts.items():
    words_list.append(list(word_tuple))   #转换为list方便后续修改
    count_list.append(freq)

  #stats 存储所有可能相邻的对及其全局词频 {(byte_a, byte_b): frequency}
  stats = defaultdict(int)

  #indices: 创建索引,存储pair -> {包含该pair 的单词 在 word_list 下的下标集合}
  indices = defaultdict(set)

  # 初始化 stats 和 indices
  #遍历一遍所有单词
  for idx, word in enumerate(words_list):
    freq = count_list[idx]
    for i in range(len(word) - 1):
      pair = (word[i], word[i+1])
      stats[pair] += freq 
      indices[pair].add(idx)
  
  merges = [] #存储生成的BPE 合并规则,按顺序记录

  #4. 迭代合并
  for _ in range(num_merges):
    if not stats:
      break

    #4.a 寻找best_pair
    best_pair = max(stats.items(),key=lambda x :(x[1], x[0]))[0] #先比较频率, 频率一致再比较字典序

    if stats[best_pair] <= 0:
      break

    #记录合并
    merges.append(best_pair)
    new_token = best_pair[0] + best_pair[1]

    #4.b 获取需要更新的单词
    #倒序索引 快速获取所有包含 "best_pair"的单词下标
    #必须复制一份"relevant_indices" 因为后面的循环会修改 "indices" 和"stats"
    relevant_indices = list(indices[best_pair])

    #4.c 遍历并更新所有受影响的单词,统计信息和倒排索引
    for idx in relevant_indices:
      word = words_list[idx]
      freq = count_list[idx]

      i = 0
      while i < len(word) - 1:
        if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
          #匹配到了

          #1. 更新旧的邻居pair的频率
          if i > 0:
            prev_pair = (word[i-1],word[i])
            stats[prev_pair] -= freq
            if stats[prev_pair] == 0:
              del stats[prev_pair] #去掉这个pair 防止 全为0的时候出错
          
          if i < len(word) - 2:
            next_pair = (word[i+1],word[i+2])
            stats[next_pair] -= freq
            if stats[next_pair] == 0:
              del stats[next_pair]

          #2. 修改单词结构
          word[i] = new_token
          del word[i+1]

          #3.添加产生的新邻居Pair的频率和索引
          if i > 0:
            new_prev = (word[i-1],word[i])
            stats[new_prev] += freq
            indices[new_prev].add(idx)

          if i < len(word) - 1:
            new_next = (word[i],word[i+1])
            stats[new_next] += freq
            indices[new_next].add(idx)
          #不需要i + 1
        else:
          i += 1
      
    #4d. 清理:移除已经合并的best_pair
    #这个pair 已经不存在在"stats" 和"indices"中了
    if best_pair in stats: del stats[best_pair]
    if best_pair in indices: del indices[best_pair]

  #5. 构建最终的词表
  #添加bpe合并产生的Token ID 从256开始
  for pair in merges:
    new_id = len(vocab)
    vocab[new_id] = pair[0] + pair[1]

  # 添加特殊token
  for s_tok in special_tokens:
    s_bytes = s_tok.encode("utf-8")
    vocab[len(vocab)] = s_bytes

  return vocab, merges

def bytes_to_unicode():
    """
    返回一个字典，将 0~255 的每个字节映射为一个唯一的 Unicode 字符。
    目的是让 BPE 处理 Unicode 字符串，避免对原始字节（尤其是不可见控制字符）进行操作，
    同时保证映射是双射的（可逆）。
    """
    # 0~255 中所有的可打印 Latin-1 字符（即 33~126 和 161~255）
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # 副本，后面会被替换成对应的字符
    n = 0
    for b in range(2 ** 8):   # 0~255
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)   # 用 256+n 作为“占位”数字，稍后转成 chr
            n += 1
    cs = [chr(n) for n in cs]       # 将数字全部转为 Unicode 字符
    return dict(zip(bs, cs))

def save_tokenizer_files(vocab, merges, out_dir):
  os.makedirs(out_dir,exist_ok = True)

  #初始化映射表
  byte_encoder = bytes_to_unicode()

  #词表保存
  json_vocab = {
    k: "".join(byte_encoder[b] for b in v)
    for k , v in vocab.items()
  }

  with open(os.path.join(out_dir,"vocab.json"),"w",encoding = "utf-8") as f:
    json.dump(json_vocab,f,indent=4)

  with open(os.path.join(out_dir, "merges.txt"),"w",encoding = "utf-8") as f:
    for p1, p2 in merges:
      s1 = "".join(byte_encoder[b] for b in p1)
      s2 = "".join(byte_encoder[b] for b in p2)
      f.write(f"{s1} {s2}\n")
 

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"  # 你的原始文本路径
    vocab_size = 10000  # 作业要求的词表大小
    # input_path = "data/owt_train.txt"
    # input_path = "data/chinese.txt"
    # vocab_size = 1000  # 作业要求的词表大小

    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器（目标词表大小: {vocab_size}）...")
    print(f"这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")

    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()




