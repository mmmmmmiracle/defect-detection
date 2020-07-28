# for model in efficientnet-b0 efficientnet-b1 efficientnet-b2 efficientnet-b3 efficientnet-b4 efficientnet-b5 efficientnet-b6 efficientnet-b7;
# do  
# python script.py -backbone ${model}
# # python inference.py -backbone ${model}
# done

for model in timm-efficientnet-b8;
do  
python script.py -backbone ${model}
python inference.py -backbone ${model}
done

# for model in xception inceptionv4;
# do  
# # python script.py -backbone ${model}
# python inference.py -backbone ${model}
# done

# for model in resnet18 resnet34 resnet50 resnet101 resnet152 resnext50_32x4d resnext101_32x8d;
# do  
# python script.py -backbone ${model}
# done

# for model in dpn68 dpn68b dpn92 dpn98 dpn107 dpn131;
# do  
# python script.py -backbone ${model}
# done

# for model in vgg11_bn vgg13_bn vgg16_bn vgg19_bn;
# do  
# # python script.py -backbone ${model}
# python inference.py -backbone ${model}
# done

# for model in senet154 se_resnet50 se_resnet101 se_resnet152 se_resnext50_32x4d se_resnext101_32x4d;
# do  
# python script.py -backbone ${model}
# done

# for model in densenet121 densenet169 densenet201 densenet161 inceptionresnetv2 inceptionv4 xception;
# do  
# python script.py -backbone ${model}
# done