# MIMO-OFDM-simulation-program-exercises

07/27

在ML detection中，一開始用了python寫出來的版本，雖然用了numpy的強大矩陣運算，但速度不是很理想

因此後來在c++重新寫了一遍，速度提升了數幾倍

以mimo 2X2為例，16QAM，N=1千萬個點，16個snr需約8.93h，而在C++版本為約3.9h


07/28

python版本ML detection為遞迴方法

修正一些小bugs，C++版本ML detection改為非遞迴方法

其中c++版本使用了第三方函式庫eigen，原本想使用armadillo+openblas配置，但研究了半天還是失敗，果斷放棄用openblas