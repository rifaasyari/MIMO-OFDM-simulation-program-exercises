﻿# MIMO-OFDM-simulation-program-exercises

07/27

在ML detection中，一開始用了python寫出來的版本，雖然用了numpy的強大矩陣運算，但速度不是很理想

因此後來在c++重新寫了一遍，速度提升了數幾倍


07/28

python版本ML detection為遞迴方法

修正一些小bugs，C++版本ML detection改為非遞迴方法

其中c++版本使用了第三方函式庫eigen，原本想使用armadillo+openblas配置，但研究了半天還是失敗，果斷放棄openblas


07/29

c++版本加入了OpenMP來做對特定程式區塊平行化，初步修改程式碼後，以我的電腦(8核)為例，在速度上提升了一些，理論上可以增快8倍，認為應該還可以再更優化程式碼

以mimo 2X2為例，16QAM，N=1000，16個snr，python版本需約34.61s，而在C++版本(沒有平行化)為約7.886s，C++版本(平行化)為約1.834s

預估完成 N = 1千萬點的時間分別為，96.19h，21.9h，5.1h。比起python版，C++版本(平行化)速度提升了約19倍

目前先上傳C++版本(平行化)，程式碼還未整理，之後幾天再慢慢完善