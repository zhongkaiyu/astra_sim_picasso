import json

with open("utilization.json", "r") as f:
    data = json.load(f)


def get_proj_qkv(bs):
    return data["proj_qkv"][str(bs)]


def get_score(seq_len):
    # score is Q @ K^T
    return data["score"][str(seq_len)]["utilization"]


def get_attention(seq_len):
    # attention is S @ V
    return data["attention"][str(seq_len)]["utilization"]


def get_proj_o(bs):
    return data["proj_o"][str(bs)]


if __name__ == "__main__":
    print(f"proj_qkv(bs=8):       {get_proj_qkv(8)}")
    print(f"score(seq=4096):      {get_score(4096)}")
    print(f"attention(seq=4096):  {get_attention(4096)}")
    print(f"proj_o(bs=8):         {get_proj_o(8)}")
