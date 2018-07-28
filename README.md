# MIMO-OFDM-simulation-program-exercises

07/27

在ML detection中，一開始用了python寫出來的版本，雖然用了numpy的強大矩陣運算，但速度不是很理想

因此後來在c++重新寫了一遍，速度提升了數幾倍

以mimo 1X1為例，16QAM，N=1千萬個點，16個snr需約50min，而在C++版本為約11min

以mimo 2X2為例，16QAM，N=1千萬個點，16個snr需約8.93h，而在C++版本為約3.9h

以mimo 4X4為例，16QAM，N=1千萬個點，時間約為...太久了

當天線數增加時，兩者版本倍差減小

由於ML算法是暴力算法，接下來構想修改程式碼，把4x4 ML暴力法完成

