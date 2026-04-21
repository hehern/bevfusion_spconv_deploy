import onnx

path_onnx = 'model/resnet50/lidar.backbone.xyz.onnx'
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
node_list = ['conv0', 'conv5', 'conv10', 'conv15', 'conv20', 'scatter0', 'transpose0', 'reshape0']
remove_nodes = []
for node in model.graph.node:
    if node.name not in node_list:
        remove_nodes.append(node)
    elif node.name == 'conv5':
        node.input[0] = '1'
    elif node.name == 'conv10':
        node.input[0] = '10'
    elif node.name == 'conv15':
        node.input[0] = '19'
    elif node.name == 'conv20':
        node.input[0] = '28'
    elif node.name == 'scatter0':
        node.input[0] = '37'
        # print(node.input[0])
        # import ipdb; ipdb.set_trace()
        for attr in node.attribute:
            if attr.name == 'input_spatial_shape':
                attr.ints[0] = 180
                attr.ints[1] = 180
                attr.ints[2] = 2
            elif attr.name == 'output_shape':
                attr.ints[0] = 1
                attr.ints[1] = 128
                attr.ints[2] = 180
                attr.ints[3] = 180
                attr.ints[4] = 2
    elif node.name == 'reshape0':
        for attr in node.attribute:
            if attr.name == 'dims':
                attr.ints[0] = 1
                attr.ints[1] = 256
                attr.ints[2] = 180
                attr.ints[3] = 180

for output in model.graph.output:
    # import ipdb; ipdb.set_trace()
    d = output.type.tensor_type.shape.dim
    print(d)
    d[0].dim_value = 1
    d[1].dim_value = 256
    d[2].dim_value = 180
    d[3].dim_value = 180
    

for item in remove_nodes[::-1]:
    model.graph.node.remove(item)

onnx.save(model, 'model/resnet50/5node.lidar.backbone.xyz.onnx')