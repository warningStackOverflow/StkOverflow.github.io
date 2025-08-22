---
layout: default
---

# 第13周实习记录
## 7月7日
### er模型线上部署
1. 重新训练了将归一化流程转为模型参数的er模型，模型保存路径为：‘./model/ESMM_50_20250707_1101_esmm_er.pth‘
2. 将模型转为tf格式并且成功部署到线上。学习了线上部署流程，回顾了一下基本命令行和processor的使用。
## 7月8日
发现线上模型部署时，模型params缺失，排查原因，发现是生成的testcase没有按照tf模型的testcase格式。修复后重新部署模型，成功上线。
从pytorch模型到tf模型的转换流程如下：
1. 将pytorch模型转为onnx模型，使用torch.onnx.export()函数。转换时要确保pytorch模型打开了eval模式，确保模型中所有的random被固定
2. 将onnx模型转为tf模型，得到tf模型的SavedModel格式的文件夹。
3. 对同一条数据，使用pytorch模型和tf模型进行验证，确保两者输出一致。
4. 生成upload_config文件，其中包含test_case, feature,model等文件
5. 按照upload_to_garden.py的要求上传模型到线上。

## 7月9日
申请线上机器，申请权限，查看数据链路等。

## 7月10日
路径在home/xiaoju/dintl-pricing/logs， 主要查看～.log.wf, 和～.log文件，这些是go中可能的报错error日志。
查看日志，发现两个主要报错，一个是get degrade fail，一个是config不存在。 <br>
[WARNING][2025-07-10T16:00:00.326+0800][..m/international-marketplace/dintl-pricing/iowrapper.doGetDegrade/degrade.go:281] _undef||ctx_format=unset||get degrade
 fail||node=DegradeDintlPricingRedisFusion||err=current config is empty <br>
[WARNING][2025-07-10T16:00:00.326+0800][..nal-marketplace/dintl-pricing/strategy/dp.(*DP7_4).SetValidUntil/dp_7_4.go:956] _undef||traceid=065603ba686f728800006b
214b9a2594||hintcode=0||product_id=16||city_id=55000224||GetBucketMetaData! Error is get bucket_meta_data fail <br>
### 处理config缺失的问题：
通过查看线上go代码，发现问题出现在process_v2.go文件中，dealDPTarget函数，其中读取surgeConfig时需要调用 iowrapper.GetSurgeConfig <br> 
其中调用语句 apolloutil.GetConfigItemValue(DINTL_PRICING_SURGE_CONFIG_V2, configName,strconv.FormatInt(cityId, 10), &surgeConfigV2); <br>
获取对应的namespace, 在这里似乎namespace[dintl_pricing_surge_config_v2]中没有对应的key或value导致报错。

## 7月11日
对线上sql数据进行分析，验证策略效果：<br> 
对两个策略dp8.0和dp8.1对er的预测情况进行分析，前者作为c组，后者作为t组。<br>
1. dp8.0策略：通过预估cr*ecr = er方式计算er。<br>
2. dp8.1策略：直接预估er。<br>
3. 对于两组的er值，分别计算和真实er间的mae等指标
