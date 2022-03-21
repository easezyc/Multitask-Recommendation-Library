# Multi-task Recommendation in PyTorch
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)  [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![MTRec](./mtreclib.png)

-------------------------------------------------------------------------------

## Introduction
MTReclib provides a PyTorch implementation of multi-task recommendation models and common datasets. Currently, we implmented 7 multi-task recommendation models to enable fair comparison and boost the development of multi-task recommendation algorithms. The currently supported algorithms include:
* SingleTask：Train one model for each task, respectively
* Shared-Bottom: It is a traditional multi-task model with a shared bottom and multiple towers.
* OMoE: [Adaptive Mixtures of Local Experts](https://ieeexplore.ieee.org/abstract/document/6797059) (Neural Computation 1991)
* MMoE: [Modeling Task Relationships in Multi-task Learning with Multi-Gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) (KDD 2018)
* PLE: [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/pdf/10.1145/3383313.3412236?casa_token=8fchWD8CHc0AAAAA:2cyP8EwkhIUlSFPRpfCGHahTddki0OEjDxfbUFMkXY5fU0FNtkvRzmYloJtLowFmL1en88FRFY4Q) (RecSys 2020 best paper)
* AITM: [Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising](https://dl.acm.org/doi/pdf/10.1145/3447548.3467071?casa_token=5YtVOYjJClUAAAAA:eVczwdynmE9dwoyElCG4da9fC5gsRiyX6zKt0_mIJF1K8NkU-SlNkGmpAu0c0EHbM3hBUe3zZc-o) (KDD 2021)
* MetaHeac: [Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising](https://easezyc.github.io/data/kdd21_metaheac.pdf) (KDD 2021)

## Datasets
* AliExpressDataset: This is a dataset gathered from real-world traffic logs of the search system in AliExpress. This dataset is collected from 5 countries: Russia, Spain, French, Netherlands, and America, which can utilized as 5 multi-task datasets. [Original_dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690) [Processed_dataset Google Drive](https://drive.google.com/drive/folders/1F0TqvMJvv-2pIeOKUw9deEtUxyYqXK6Y?usp=sharing) [Processed_dataset Baidu Netdisk](https://pan.baidu.com/s/1AfXoJSshjW-PILXZ6O19FA?pwd=4u0r)

> For the processed dataset, you should directly put the dataset in './data/' and unpack it. For the original dataset, you should put it in './data/' and run 'python preprocess.py --dataset_name NL'.

## Requirements
* Python 3.6
* PyTorch > 1.10
* pandas
* numpy
* tqdm


## Run

Parameter Configuration:

- dataset_name: choose a dataset in ['AliExpress_NL', 'AliExpress_FR', 'AliExpress_ES', 'AliExpress_US'], default for `AliExpress_NL`
- dataset_path: default for `./data`
- model_name: choose a model in ['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac'], default for `metaheac`
- epoch: the number of epochs for training, default for `50`
- task_num: the number of tasks, default for `2` (CTR & CVR)
- expert_num: the number of experts for ['omoe', 'mmoe', 'ple', 'metaheac'], default for `8`
- learning_rate: default for `0.001`
- batch_size: default for `2048`
- weight_decay: default for `1e-6`
- device: the device to run the code, default for `cuda:0`
- save_dir: the folder to save parameters, default for `chkpt`

You can run a model through:

```powershell
python main.py --model_name metaheac --num_expert 8 --dataset_name AliExpress_NL
```

## Results
> For fair comparisons, the learning rate is 0.001, the dimension of embeddings is 128, and mini-batch size is 2048 equally for all models. We report the mean AUC and Logloss over five random runs. Best results are in boldface.

<table>
	<head >
		<tr>
      <th rowspan="3"; center>Methods</th>
			<th colspan="4"><center>AliExpress (Netherlands, NL)</center></th>
			<th colspan="4"><center>AliExpress (Spain, ES)</center></th>
		</tr>
		<tr >
			<th colspan="2"><center>CTR</center></th>
      <th colspan="2"><center>CTCVR</center></th>
	  <th colspan="2"><center>CTR</center></th>
      <th colspan="2"><center>CTCVR</center></th>
		</tr>
		<tr>
			<th >AUC</th>
			<th >Logloss</th>
			<th >AUC</th>
      <th >Logloss</th>
	  <th >AUC</th>
			<th >Logloss</th>
			<th >AUC</th>
      <th >Logloss</th>
		</tr>
	</head>
	<body>
		<tr>
			<td>SingleTask</td>
			<td>0.7222 </td>
			<td>0.1085</td>
			<td>0.8590</td>
      <td>0.00609</td>
	  <td>0.7266</td>
			<td>0.1207</td>
			<td>0.8855</td>
      <td>0.00456</td>
		</tr>
    <tr>
			<td>Shared-Bottom</td>
			<td>0.7228</td>
			<td>0.1083</td>
			<td>0.8511</td>
      <td>0.00620</td>
	  <td>0.7287</td>
			<td>0.1204</td>
			<td>0.8866</td>
      <td>0.00452</td>
		</tr>
    <tr>
			<td>OMoE</td>
			<td>0.7254</td>
			<td>0.1081</td>
			<td>0.8611</td>
      <td>0.00614</td>
	  <td>0.7253</td>
			<td>0.1209</td>
			<td>0.8859</td>
      <td>0.00452</td>
		</tr>
    <tr>
			<td>MMoE</td>
			<td>0.7234</td>
			<td>0.1080</td>
			<td>0.8606</td>
      <td>0.00607</td>
	  <td>0.7285</td>
			<td>0.1205</td>
			<td>0.8898</td>
      <td><strong>0.00450</strong></td>
		</tr>
    <tr>
			<td>PLE</td>
			<td><strong>0.7292</strong></td>
			<td>0.1088</td>
			<td>0.8591</td>
      <td>0.00631</td>
			<td>0.7273</td>
			<td>0.1223</td>
			<td><strong>0.8913</strong></td>
      <td>0.00461</td>
		</tr>
    <tr>
			<td>AITM</td>
			<td>0.7240</td>
			<td>0.1078</td>
			<td>0.8577</td>
      <td>0.00611</td>
	  <td>0.7290</td>
			<td><strong>0.1203</strong></td>
			<td>0.8885</td>
      <td>0.00451</td>
		</tr>
    <tr>
			<td>MetaHeac</td>
			<td>0.7263</td>
			<td><strong>0.1077</strong></td>
			<td><strong>0.8615</strong></td>
      <td><strong>0.00606</strong></td>
	  <td><strong>0.7299</strong></td>
			<td><strong>0.1203</strong></td>
			<td>0.8883</td>
      <td><strong>0.00450</strong></td>
		</tr>
	</body>
</table>

<table>
	<head >
		<tr>
      <th rowspan="3"; center>Methods</th>
			<th colspan="4"><center>AliExpress (French, FR)</center></th>
			<th colspan="4"><center>AliExpress (America, US)</center></th>
		</tr>
		<tr >
			<th colspan="2"><center>CTR</center></th>
      <th colspan="2"><center>CTCVR</center></th>
	  <th colspan="2"><center>CTR</center></th>
      <th colspan="2"><center>CTCVR</center></th>
		</tr>
		<tr>
			<th >AUC</th>
			<th >Logloss</th>
			<th >AUC</th>
      <th >Logloss</th>
	  <th >AUC</th>
			<th >Logloss</th>
			<th >AUC</th>
      <th >Logloss</th>
		</tr>
	</head>
	<body>
		<tr>
			<td>SingleTask</td>
			<td>0.7259</td>
			<td><strong>0.1002</strong></td>
			<td>0.8737</td>
      <td>0.00435</td>
	  <td>0.7061</td>
			<td>0.1004</td>
			<td>0.8637</td>
      <td>0.00381</td>
		</tr>
    <tr>
			<td>Shared-Bottom</td>
			<td>0.7245</td>
			<td>0.1004</td>
			<td>0.8700</td>
      <td>0.00439</td>
	  <td>0.7029</td>
			<td>0.1008</td>
			<td>0.8698</td>
      <td>0.00381</td>
		</tr>
    <tr>
			<td>OMoE</td>
			<td>0.7257</td>
			<td>0.1006</td>
			<td>0.8781</td>
      <td>0.00432</td>
	  <td>0.7049</td>
			<td>0.1007</td>
			<td>0.8701</td>
      <td>0.00381</td>
		</tr>
    <tr>
			<td>MMoE</td>
			<td>0.7216</td>
			<td>0.1010</td>
			<td>0.8811</td>
      <td>0.00431</td>
	  <td>0.7043</td>
			<td>0.1006</td>
			<td><strong>0.8758</strong></td>
      <td><strong>0.00377</strong></td>
		</tr>
    <tr>
			<td>PLE</td>
			<td><strong>0.7276</strong></td>
			<td>0.1014</td>
			<td>0.8805</td>
      <td>0.00451</td>
			<td><strong>0.7138</strong></td>
			<td><strong>0.0992</strong></td>
			<td>0.8675</td>
      <td>0.00403</td>
		</tr>
    <tr>
			<td>AITM</td>
			<td>0.7236</td>
			<td>0.1005</td>
			<td>0.8763</td>
      <td>0.00431</td>
	  <td>0.7048</td>
			<td>0.1004</td>
			<td>0.8730</td>
      <td><strong>0.00377</strong></td>
		</tr>
    <tr>
			<td>MetaHeac</td>
			<td>0.7249</td>
			<td>0.1005</td>
			<td><strong>0.8813</strong></td>
      <td><strong>0.00429</strong></td>
	  <td>0.7089</td>
			<td>0.1001</td>
			<td>0.8743</td>
      <td>0.00378</td>
		</tr>
	</body>
</table>

## File Structure

```
.
├── main.py
├── README.md
├── models
│   ├── layers.py
│   ├── aitm.py
│   ├── omoe.py
│   ├── mmoe.py
│   ├── metaheac.py
│   ├── ple.py
│   ├── singletask.py
│   └── sharedbottom.py
└── data
    ├── preprocess.py         # Preprocess the original data
    ├── AliExpress_NL         # AliExpressDataset from Netherlands
    	├── train.csv
	└── test.py
    ├── AliExpress_ES         # AliExpressDataset from Spain
    ├── AliExpress_FR         # AliExpressDataset from French
    └── AliExpress_US         # AliExpressDataset from America
```



## Contact
If you have any problem about this library, please create an issue or send us an Email at:
* zhuyc0204@gmail.com


## Reference
If you use this repository, please cite the following papers:

```
@inproceedings{zhu2021learning,
  title={Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising},
  author={Zhu, Yongchun and Liu, Yudan and Xie, Ruobing and Zhuang, Fuzhen and Hao, Xiaobo and Ge, Kaikai and Zhang, Xu and Lin, Leyu and Cao, Juan},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={4005--4013},
  year={2021}
}
```

```
@inproceedings{xi2021modeling,
  title={Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising},
  author={Xi, Dongbo and Chen, Zhen and Yan, Peng and Zhang, Yinger and Zhu, Yongchun and Zhuang, Fuzhen and Chen, Yu},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={3745--3755},
  year={2021}
}
```

Some model implementations and util functions refers to these nice repositories.

- [pytorch-fm](https://github.com/rixwew/pytorch-fm): This package provides a PyTorch implementation of factorization machine models and common datasets in CTR prediction. 
- [MetaHeac](https://github.com/easezyc/MetaHeac): This is an official implementation for Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising.
