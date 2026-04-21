import onnx

path_onnx = 'model/resnet50/5node.lidar.backbone.xyz.onnx'
model = onnx.load(path_onnx)
# import ipdb; ipdb.set_trace()

# 所有节点的输出都引出来
# for node in model.graph.node:
#     for output in node.output:
#         model.graph.output.extend([onnx.ValueInfoProto(name=output)])
# onnx.save(model, 'model/resnet50/mulout.lidar.backbone.xyz.onnx') 

# 把特定的几个node的输出引出来
# node_list = ['conv0', 'relu0']
# for node in model.graph.node:
#     if node.name in node_list:
#         for output in node.output:
#             model.graph.output.extend([onnx.ValueInfoProto(name=output)])
# onnx.save(model, 'model/resnet50/mulnode.lidar.backbone.xyz.onnx') 

# 删除nodes
node_list = ['conv0', 'scatter0', 'transpose0', 'reshape0']
remove_nodes = []
for node in model.graph.node:
    if node.name not in node_list:
        remove_nodes.append(node)
    elif node.name == 'scatter0':
        node.input[0] = '1'
        # print(node.input[0])
        # import ipdb; ipdb.set_trace()
        for attr in node.attribute:
            if attr.name == 'input_spatial_shape':
                attr.ints[0] = 1440
                attr.ints[1] = 1440
                attr.ints[2] = 41
            elif attr.name == 'output_shape':
                attr.ints[0] = 1
                attr.ints[1] = 16
                attr.ints[2] = 1440
                attr.ints[3] = 1440
                attr.ints[4] = 41
    elif node.name == 'reshape0':
        for attr in node.attribute:
            if attr.name == 'dims':
                attr.ints[0] = 1
                attr.ints[1] = 656
                attr.ints[2] = 1440
                attr.ints[3] = 1440

for output in model.graph.output:
    # import ipdb; ipdb.set_trace()
    d = output.type.tensor_type.shape.dim
    print(d)
    d[0].dim_value = 1
    d[1].dim_value = 656
    d[2].dim_value = 1440
    d[3].dim_value = 1440
    

for item in remove_nodes[::-1]:
    model.graph.node.remove(item)

onnx.save(model, 'model/resnet50/singlenode.lidar.backbone.xyz.onnx')