This is a repository for implementations of 
- Sequence to Sequence [arXiv](https://arxiv.org/abs/1409.3215)
- Pointer Networks [arXiv](https://arxiv.org/abs/1506.03134)

in chainer.

---

### Todo
- 論文実験の再現
  - ハイパラ、Optimizerの調節
  - 論文で用いたデータセットを利用した実験の実施
  - Seq2Seqモデルでも同じことする

- Seq2Seq部分のリファクタリング
  - Decoderの構成を少しきちんと考えたほうがよいと思います（それはそう）
  
- Pointer Networks部分のリファクタリング
  - for文を使って回している部分があるのでなんとかできたら嬉しい（がつらそう）
  - 様々なサイズの入力が可能かどうかを試してみたい（多分無理なので、可変長系列対応ver.にする）

### Links
- Pointer Networksのデータセット
  - https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
