This is a repository for implementations of 
- Sequence to Sequence [arXiv](https://arxiv.org/abs/1409.3215)
- Pointer Networks [arXiv](https://arxiv.org/abs/1506.03134)

in chainer.

### Dependencies
I was running this code under 

- python 3.7
- chainer 
- hoge
- huga

I included requirements.txt in repository, so you can install all dependencies by running below script.

```
pip install -r requirements.
```

### How to use
Under construction...

### (Optional) How to prepare dataset
Under construction...



---

### Todo
- 論文実験の再現
  - 論文で用いたデータセットを利用した実験の実施
    - beam searchをするにはEncoderとDecoder部分を分けないと辛いんだよなぁ……悲しみ
  - Seq2Seqモデルでも同じことする

- Seq2Seq部分のリファクタリング
  - Decoderの構成を少しきちんと考えたほうがよいと思います（それはそう）
  
- Pointer Networks部分のリファクタリング
  - for文を使って回している部分があるのでなんとかできたら嬉しい（が無理そう）
    - Attentionクラスの、__call__部分をbatchごとにできたら嬉しい……んだけど無理だよなぁ

### Links
- Pointer Networksのデータセット
  - https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
