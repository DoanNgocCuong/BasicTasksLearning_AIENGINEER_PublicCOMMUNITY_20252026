1. Tính toán số lần xuất hiện của từng nhãn: => lấy top 17 (a Nam)
2. Tính toán số nhãn (cả đơn cả đa) => lấy top 30 nhãn (cả đơn, cả đa) mà cover hết 90% data

---

Cả 2 cách trên: 

=> sau train đều gặp vấn đề là: nhãn con bị mất cân bằng.

Xem file: json_categories_statistic.xlsx

---



Bài toán đặt ra là: 

1. Cân bằng nhãn cha

 ta cần **chọn một tập nhãn_cha** sao cho:

- Tổng (coverage) của tập này ≥ ngưỡng τ (80–90% tổng data), **và**

- Tập này **cân bằng** (không quá lệch).

2. Cân bằng nhãn con:

   tương tự:

ta cần **chọn một tập nhãn_ccon** sao cho:

- Tổng (coverage) của tập này ≥ ngưỡng τ (80–90% tổng data của từng nhãn cha), **và**
- Tập này **cân bằng** (không quá lệch).


Thuật toán đề xuất: 

1. Tính 85% của tổng, lấy các số từ lớn đến bé sao cho tổng của nó lớn hơn 85% thì dừng
2. Với tập data con đó thì lấy R = 2, bỏ đi những thằng con nào mà bị nhỏ hơn 2 lần thằng max

   (chấp nhận việc ảnh hưởng tới 85%)

   ---

   Cách 3:

   Thay vì việc phải chọn xem hay giữ lại những thằng con nhỏ hơn 2 lần max
   => Thay bằng việc với mỗi thằng con:


   1. Tính tổng 85%
   2. Thay vì lấy các nhãn con nhỏ hơn 2 lần max. Thay bằng việc mỗi nhãn lấy số lượng = số lượng của thằng con có số lượng nhãn nhỏ nhất.


| nhãn_cha | nhãn_con                                            | số_lần_xuất_hiện_nhãn_con |     |       |              |
| --------- | ---------------------------------------------------- | ------------------------------ | --- | ----- | ------------ |
|           |                                                      | #REF!                          |     |       |              |
| hep-ph    | hep-ph                                               | 10000                          | 1   |       |              |
| hep-th    | hep-th                                               | 10000                          | 1   |       |              |
| gr-qc     | gr-qc                                                | 8000                           | 3   |       |              |
| quant-ph  | quant-ph                                             | 6000                           | 4   |       |              |
| hep-ex    | hep-ex                                               | 4000                           | 5   |       |              |
| nucl-th   | nucl-th                                              | 4000                           | 5   |       |              |
| astro-ph  | [](http://astro-ph.CO)[astro-ph.CO](http://astro-ph.CO) | 2416                           | 7   |       |              |
| math      | math-ph                                              | 2070                           | 8   |       |              |
| math      | [](http://math.MP)[math.MP](http://math.MP)             | 2070                           | 8   |       |              |
| hep-lat   | hep-lat                                              | 2000                           | 10  |       |              |
| math-ph   | math-ph                                              | 2000                           | 10  |       |              |
| nucl-ex   | nucl-ex                                              | 2000                           | 10  |       |              |
| cond-mat  | cond-mat.stat-mech                                   | 1535                           | 13  |       |              |
| cond-mat  | cond-mat.str-el                                      | 1523                           | 14  |       |              |
| astro-ph  | astro-ph.HE                                          | 1462                           | 15  |       |              |
| cond-mat  | cond-mat.mes-hall                                    | 1381                           | 16  |       |              |
| physics   | physics.optics                                       | 1351                           | 17  |       |              |
| astro-ph  | astro-ph                                             | 1330                           | 18  |       |              |
| stat      | [](http://stat.ME)[stat.ME](http://stat.ME)             | 870                            | 19  |       |              |
| econ      | [](http://econ.GN)[econ.GN](http://econ.GN)             | 862                            | 20  |       |              |
| stat      | [](http://stat.ML)[stat.ML](http://stat.ML)             | 828                            | 21  |       |              |
| nlin      | [](http://nlin.CD)[nlin.CD](http://nlin.CD)             | 760                            | 22  |       |              |
| eess      | eess.SP                                              | 733                            | 23  |       |              |
| econ      | econ.EM                                              | 719                            | 24  |       |              |
| physics   | physics.atom-ph                                      | 716                            | 25  |       |              |
| cond-mat  | cond-mat.mtrl-sci                                    | 641                            | 26  |       |              |
| eess      | eess.IV                                              | 586                            | 27  |       |              |
| nlin      | [](http://nlin.SI)[nlin.SI](http://nlin.SI)             | 576                            | 28  |       |              |
| q-bio     | [](http://q-bio.PE)[q-bio.PE](http://q-bio.PE)          | 570                            | 29  |       |              |
| cs        | cs.LG                                                | 551                            | 30  |       |              |
| stat      | stat.AP                                              | 529                            | 31  |       |              |
| eess      | [](http://eess.SY)[eess.SY](http://eess.SY)             | 522                            | 32  |       |              |
| astro-ph  | [](http://astro-ph.GA)[astro-ph.GA](http://astro-ph.GA) | 519                            | 33  |       |              |
| astro-ph  | [](http://astro-ph.SR)[astro-ph.SR](http://astro-ph.SR) | 512                            | 34  |       |              |
| nlin      | [](http://nlin.PS)[nlin.PS](http://nlin.PS)             | 510                            | 35  |       |              |
| econ      | [](http://econ.TH)[econ.TH](http://econ.TH)             | 492                            | 36  |       |              |
| cond-mat  | cond-mat.quant-gas                                   | 486                            | 37  |       |              |
| cond-mat  | cond-mat.supr-con                                    | 486                            | 37  |       |              |
| q-fin     | [](http://q-fin.ST)[q-fin.ST](http://q-fin.ST)          | 477                            | 39  |       |              |
| q-bio     | q-bio.QM                                             | 445                            | 40  | 76528 | 0.835203213  |
| q-bio     | [](http://q-bio.NC)[q-bio.NC](http://q-bio.NC)          | 422                            | 41  | 76950 |              |
| cs        | [](http://cs.CV)[cs.CV](http://cs.CV)                   | 400                            | 42  |       |              |
| cond-mat  | cond-mat.other                                       | 384                            | 43  |       |              |
| math      | math.AP                                              | 376                            | 44  |       |              |
| q-fin     | [](http://q-fin.MF)[q-fin.MF](http://q-fin.MF)          | 359                            | 45  |       |              |
| physics   | physics.chem-ph                                      | 357                            | 46  |       |              |
| q-fin     | q-fin.RM                                             | 321                            | 47  |       |              |
| nlin      | [](http://nlin.AO)[nlin.AO](http://nlin.AO)             | 317                            | 48  |       |              |
| q-fin     | [](http://q-fin.GN)[q-fin.GN](http://q-fin.GN)          | 317                            | 48  |       |              |
| q-fin     | q-fin.CP                                             | 314                            | 50  |       |              |
| q-fin     | [](http://q-fin.PR)[q-fin.PR](http://q-fin.PR)          | 305                            | 51  | 80400 | 0.8774610381 |
| cs        | [](http://cs.AI)[cs.AI](http://cs.AI)                   | 302                            | 52  |       |              |
| math      | [](http://math.PR)[math.PR](http://math.PR)             | 296                            | 53  |       |              |
| cond-mat  | cond-mat.dis-nn                                      | 290                            | 54  |       |              |
| astro-ph  | [](http://astro-ph.IM)[astro-ph.IM](http://astro-ph.IM) | 282                            | 55  |       |              |
| physics   | physics.ins-det                                      | 279                            | 56  |       |              |
| math      | [](http://math.CO)[math.CO](http://math.CO)             | 275                            | 57  |       |              |
| eess      | [](http://eess.AS)[eess.AS](http://eess.AS)             | 266                            | 58  |       |              |
| q-fin     | [](http://q-fin.PM)[q-fin.PM](http://q-fin.PM)          | 264                            | 59  |       |              |
| cond-mat  | cond-mat.soft                                        | 261                            | 60  |       |              |
| physics   | physics.comp-ph                                      | 260                            | 61  |       |              |
| cond-mat  | cond-mat                                             | 250                            | 62  |       |              |
| q-bio     | [](http://q-bio.BM)[q-bio.BM](http://q-bio.BM)          | 241                            | 63  |       |              |
| stat      | [](http://stat.CO)[stat.CO](http://stat.CO)             | 235                            | 64  |       |              |
| physics   | physics.flu-dyn                                      | 234                            | 65  |       |              |
| math      | [](http://math.AG)[math.AG](http://math.AG)             | 229                            | 66  |       |              |
| cs        | [](http://cs.CL)[cs.CL](http://cs.CL)                   | 222                            | 67  |       |              |
| physics   | physics.app-ph                                       | 218                            | 68  |       |              |
| q-fin     | [](http://q-fin.TR)[q-fin.TR](http://q-fin.TR)          | 218                            | 68  |       |              |
| math      | math.DG                                              | 215                            | 70  |       |              |
| physics   | physics.hist-ph                                      | 215                            | 70  |       |              |
| astro-ph  | astro-ph.EP                                          | 208                            | 72  |       |              |
| q-bio     | [](http://q-bio.MN)[q-bio.MN](http://q-bio.MN)          | 199                            | 73  |       |              |
| math      | math.DS                                              | 188                            | 74  |       |              |
| cs        | [](http://cs.IT)[cs.IT](http://cs.IT)                   | 173                            | 75  |       |              |
| math      | math.NT                                              | 170                            | 76  |       |              |
| physics   | physics.plasm-ph                                     | 167                            | 77  |       |              |
| q-bio     | [](http://q-bio.GN)[q-bio.GN](http://q-bio.GN)          | 167                            | 77  |       |              |
| physics   | physics.soc-ph                                       | 157                            | 79  |       |              |
| math      | math.FA                                              | 156                            | 80  |       |              |
| math      | math.OC                                              | 153                            | 81  |       |              |
| math      | math.RT                                              | 143                            | 82  |       |              |
| math      | [](http://math.NA)[math.NA](http://math.NA)             | 132                            | 83  |       |              |
| physics   | physics.bio-ph                                       | 132                            | 83  |       |              |
| cs        | [](http://cs.CR)[cs.CR](http://cs.CR)                   | 125                            | 85  |       |              |
| math      | math.SP                                              | 121                            | 86  |       |              |
| physics   | physics.gen-ph                                       | 121                            | 86  |       |              |
| q-fin     | [](http://q-fin.EC)[q-fin.EC](http://q-fin.EC)          | 114                            | 88  |       |              |
| physics   | physics.class-ph                                     | 112                            | 89  |       |              |
| stat      | [](http://stat.TH)[stat.TH](http://stat.TH)             | 112                            | 89  |       |              |
| math      | [](http://math.CA)[math.CA](http://math.CA)             | 109                            | 91  |       |              |
| q-bio     | [](http://q-bio.TO)[q-bio.TO](http://q-bio.TO)          | 109                            | 91  |       |              |
| q-bio     | q-bio.CB                                             | 106                            | 93  |       |              |
| physics   | physics.data-an                                      | 98                             | 94  |       |              |
| cs        | [](http://cs.RO)[cs.RO](http://cs.RO)                   | 97                             | 95  |       |              |
| math      | math.RA                                              | 94                             | 96  |       |              |
| math      | [](http://math.GR)[math.GR](http://math.GR)             | 94                             | 96  |       |              |
| math      | [](http://math.QA)[math.QA](http://math.QA)             | 93                             | 98  |       |              |
| cs        | cs.DS                                                | 91                             | 99  |       |              |
| math      | [](http://math.GT)[math.GT](http://math.GT)             | 86                             | 100 |       |              |
| math      | math.OA                                              | 84                             | 101 |       |              |
| math      | [](http://math.CV)[math.CV](http://math.CV)             | 83                             | 102 |       |              |
| q-bio     | [](http://q-bio.SC)[q-bio.SC](http://q-bio.SC)          | 81                             | 103 |       |              |
| cs        | [](http://cs.NI)[cs.NI](http://cs.NI)                   | 78                             | 104 |       |              |
| cs        | cs.DC                                                | 77                             | 105 |       |              |
| q-bio     | q-bio.OT                                             | 75                             | 106 |       |              |
| physics   | physics.acc-ph                                       | 74                             | 107 |       |              |
| nlin      | [](http://nlin.CG)[nlin.CG](http://nlin.CG)             | 69                             | 108 |       |              |
| cs        | [](http://cs.IR)[cs.IR](http://cs.IR)                   | 67                             | 109 |       |              |
| math      | [](http://math.SG)[math.SG](http://math.SG)             | 67                             | 109 |       |              |
| cs        | cs.LO                                                | 64                             | 111 |       |              |
| math      | [](http://math.ST)[math.ST](http://math.ST)             | 64                             | 111 |       |              |
| math      | math.LO                                              | 63                             | 113 |       |              |
| physics   | physics.med-ph                                       | 62                             | 114 |       |              |
| math      | [](http://math.AT)[math.AT](http://math.AT)             | 61                             | 115 |       |              |
| cs        | [](http://cs.CY)[cs.CY](http://cs.CY)                   | 59                             | 116 |       |              |
| cs        | [](http://cs.SI)[cs.SI](http://cs.SI)                   | 59                             | 116 |       |              |
| physics   | physics.ao-ph                                        | 56                             | 118 |       |              |
| math      | math.AC                                              | 55                             | 119 |       |              |
| physics   | physics.geo-ph                                       | 55                             | 119 |       |              |
| cs        | [](http://cs.SE)[cs.SE](http://cs.SE)                   | 54                             | 121 |       |              |
| physics   | physics.ed-ph                                        | 54                             | 121 |       |              |
| math      | [](http://math.MG)[math.MG](http://math.MG)             | 53                             | 123 |       |              |
| cs        | [](http://cs.NE)[cs.NE](http://cs.NE)                   | 51                             | 124 |       |              |
| cs        | cs.HC                                                | 50                             | 125 |       |              |
| cs        | [](http://cs.DM)[cs.DM](http://cs.DM)                   | 49                             | 126 |       |              |
| physics   | physics.space-ph                                     | 47                             | 127 |       |              |
| cs        | [](http://cs.SY)[cs.SY](http://cs.SY)                   | 44                             | 128 |       |              |
| physics   | physics.atm-clus                                     | 43                             | 129 |       |              |
| stat      | stat.OT                                              | 43                             | 129 |       |              |
| cs        | [](http://cs.GT)[cs.GT](http://cs.GT)                   | 36                             | 131 |       |              |
| math      | math.CT                                              | 33                             | 132 |       |              |
| physics   | physics.pop-ph                                       | 33                             | 132 |       |              |
| cs        | [](http://cs.SD)[cs.SD](http://cs.SD)                   | 31                             | 134 |       |              |
| cs        | cs.DB                                                | 30                             | 135 |       |              |
| cs        | [](http://cs.CC)[cs.CC](http://cs.CC)                   | 29                             | 136 |       |              |
| math      | [](http://math.GM)[math.GM](http://math.GM)             | 28                             | 137 |       |              |
| cs        | [](http://cs.PL)[cs.PL](http://cs.PL)                   | 25                             | 138 |       |              |
| math      | math.KT                                              | 24                             | 139 |       |              |
| cs        | [](http://cs.CG)[cs.CG](http://cs.CG)                   | 23                             | 140 |       |              |
| cs        | cs.DL                                                | 22                             | 141 |       |              |
| cs        | cs.FL                                                | 20                             | 142 |       |              |
| cs        | cs.CE                                                | 20                             | 142 |       |              |
| cs        | [](http://cs.AR)[cs.AR](http://cs.AR)                   | 17                             | 144 |       |              |
| cs        | [](http://cs.MA)[cs.MA](http://cs.MA)                   | 16                             | 145 |       |              |
| math      | [](http://math.GN)[math.GN](http://math.GN)             | 15                             | 146 |       |              |
| cs        | [](http://cs.NA)[cs.NA](http://cs.NA)                   | 14                             | 147 |       |              |
| math      | math.HO                                              | 12                             | 148 |       |              |
| cs        | [](http://cs.SC)[cs.SC](http://cs.SC)                   | 11                             | 149 |       |              |
| cs        | [](http://cs.GR)[cs.GR](http://cs.GR)                   | 11                             | 149 |       |              |
| cs        | [](http://cs.PF)[cs.PF](http://cs.PF)                   | 11                             | 149 |       |              |
| math      | [](http://math.IT)[math.IT](http://math.IT)             | 11                             | 149 |       |              |
| cs        | [](http://cs.MM)[cs.MM](http://cs.MM)                   | 9                              | 153 |       |              |
| cs        | [](http://cs.ET)[cs.ET](http://cs.ET)                   | 9                              | 153 |       |              |
| cs        | cs.OH                                                | 8                              | 155 |       |              |
| cs        | [](http://cs.MS)[cs.MS](http://cs.MS)                   | 7                              | 156 |       |              |
| cs        | cs.OS                                                | 3                              | 157 |       |              |
|           |                                                      | 91628                          |     |       |              |


---

=> Tập ban đầu:

- lọc nhãn cha thì lấy 17-30 thằng cha cao nhất mà cover 90% data
- , mỗi thằng cha lấy 2000 dòng data.

Tập con:

- 50 nhãn con cao nhất (theo tính toán nó chiếm tới 87% dữ liệu),
- mỗi thằng con lấy 500 dòng,


---

# Sau khi lấy 300 nhãn => thống kê số lượng mỗi dòng lại xảy ra vấn đề 

```
         nhãn  số_lần_xuất_hiện
0    quant-ph              1457
1      hep-th               994
2      hep-ph               907
3       gr-qc               722
4       cs.LG               709
..        ...               ...
153     cs.PF                 1
154  chao-dyn                 1
155     cs.OH                 1
156   math.KT                 1
157   math.AC                 1

[158 rows x 2 columns]
```


Số nhãn con lại bị 1 số cái là 1. 


---

# Lọc bỏ hết nhãn con: đi qua từng dòng data và bỏ đi nhãn con nằm ngoài list50 


```
✅ Split: train=8991, test=2249
🔄 Huấn luyện KNN...
✅ Train xong (thời gian 0.05s)
🎯 Đánh giá KNN → Accuracy=0.3433, F1-micro=0.6147, F1-macro=0.5653, Jaccard=0.5227
📊 Ví dụ 5 sample đầu tiên trong predict_df:
                                              authors  \
2              Jiliang Jing, Qiyuan Pan, Songbai Chen   
4                          A. Hopper and P. F. Barker   
6   Christian Majenz, Maris Ozols, Christian Schaf...   
9                  Alexander Grant, Eanna E. Flanagan   
11                                   Kaushik Majumdar   

                                                title  \
2   Holographic Superconductors with Power-Maxwell...   
4   A hybrid quantum system formed by trapping ato...   
6             Local simultaneous state discrimination   
9   Non-conservation of Carter in black hole space...   
11  Fourier Uniformity: An Useful Tool for Analyzi...   

                                    true_labels  \
2                               (gr-qc, hep-th)   
4   (physics.atom-ph, physics.optics, quant-ph)   
6                                   (quant-ph,)   
9                                      (gr-qc,)   
11                         (q-bio.NC, q-bio.QM)   

                               predicted_labels  
2            (cond-mat.supr-con, gr-qc, hep-th)  
4   (physics.atom-ph, physics.optics, quant-ph)  
6                                   (quant-ph,)  
9                               (gr-qc, hep-th)  
11                                  (q-bio.NC,)  
💾 Đã lưu kết quả dự đoán vào file: /content/drive/MyDrive/AIO/predict_results.csv

📊 Classification Report chi tiết cho từng nhãn:
                    precision    recall  f1-score   support

          astro-ph       0.39      0.15      0.21        61
       astro-ph.CO       0.57      0.42      0.48        77
       astro-ph.GA       0.63      0.52      0.57        60
       astro-ph.HE       0.61      0.46      0.53        71
       astro-ph.SR       0.59      0.40      0.48        60
 cond-mat.mes-hall       0.57      0.37      0.45        78
 cond-mat.mtrl-sci       0.82      0.48      0.60        65
    cond-mat.other       0.12      0.02      0.03        60
cond-mat.quant-gas       0.59      0.44      0.50        61
cond-mat.stat-mech       0.47      0.18      0.26        89
   cond-mat.str-el       0.39      0.31      0.34        81
 cond-mat.supr-con       0.75      0.73      0.74        60
             cs.AI       0.77      0.33      0.47        60
             cs.CV       0.82      0.72      0.77        76
             cs.LG       0.62      0.54      0.58       142
             cs.SY       0.80      0.58      0.67        69
           econ.EM       0.69      0.52      0.59        63
           econ.GN       0.90      0.42      0.57        64
           econ.TH       0.85      0.73      0.79        60
           eess.IV       0.87      0.79      0.83        61
           eess.SP       0.74      0.51      0.60        67
           eess.SY       0.74      0.56      0.64        61
             gr-qc       0.86      0.85      0.86       144
            hep-ex       0.83      0.56      0.67        70
           hep-lat       0.82      0.81      0.82        63
            hep-ph       0.90      0.69      0.78       181
            hep-th       0.72      0.71      0.71       199
           math-ph       0.64      0.34      0.45       125
           math.AP       0.81      0.58      0.68        60
           math.MP       0.64      0.34      0.45       125
           math.OC       0.57      0.35      0.43        60
           math.PR       0.72      0.48      0.57        61
           nlin.CD       0.84      0.54      0.66        68
           nlin.PS       0.77      0.65      0.71        63
           nlin.SI       0.75      0.74      0.74        65
           nucl-ex       0.67      0.79      0.72        61
           nucl-th       0.76      0.43      0.55        68
   physics.atom-ph       0.65      0.46      0.54        69
   physics.chem-ph       0.71      0.33      0.45        60
    physics.optics       0.63      0.43      0.51        84
    physics.soc-ph       0.42      0.16      0.24        61
          q-bio.NC       0.81      0.83      0.82        60
          q-bio.PE       0.72      0.82      0.77        62
          q-bio.QM       0.65      0.25      0.36        60
          q-fin.EC       0.81      0.42      0.56        71
          q-fin.ST       0.80      0.73      0.77        60
          quant-ph       0.89      0.87      0.88       291
           stat.AP       0.38      0.05      0.08        66
           stat.ME       0.54      0.35      0.43        74
           stat.ML       0.53      0.28      0.37        82

         micro avg       0.73      0.53      0.61      4059
         macro avg       0.68      0.50      0.57      4059
      weighted avg       0.70      0.53      0.59      4059
       samples avg       0.65      0.57      0.58      4059

```
