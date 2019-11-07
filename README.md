# Pointer-Networks
An implementation of Pointer Networks in chainer

# Todo
- ~~データを扱うためのクラス定義と、実データを読み込むまでをやる（やるだけ）~~
  - ~~データは ((x1, x2, ..., xn), (t1, t2, t3……))　みたいな感じ　もちろんnは可変~~
- Pointer Network そのものの実装（たいへん）
  - Seq2Seq部分の実装
    - Call部分のBatch化
  - Attention部分の実装
    - よくわからんので調べる
  - PointerNetworkの実装
    - わからんので調べる
- 実際に回してみる（やるだけ）

# Todo (後回し）
- Seq2Seq部分の、Decoderの構成を少しきちんと考えたほうがよいと思います（それはそう）

# Links
- Pointer Networksのデータセット
  - https://drive.google.com/drive/folders/0B2fg8yPGn2TCMzBtS0o4Q2RJaEU
