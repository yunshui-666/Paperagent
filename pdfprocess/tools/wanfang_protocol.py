"""万方官网 gRPC-Web 检索的最小 Protobuf 适配，只读取已使用的论文元数据字段。

字段依据 2026-11-19 官网公开前端的 SearchService 声明核对。
省略的 Protobuf 字段会由标准解析器跳过，站点变更时集中更新这里。
"""

import struct

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _messages():
    """构造最小消息描述，复用 protobuf 处理编码、未知字段和类型检查。"""
    schema = descriptor_pb2.FileDescriptorProto(name="paperagent_wanfang.proto", package="paperagent_wanfang")
    definitions = {
        "CommonRequest": [("search_type",1,9,False,None),("search_word",2,9,False,None),
                          ("current_page",5,5,False,None),("page_size",6,5,False,None)],
        "SearchRequest": [("common",1,11,False,"CommonRequest"),("interface_type",2,5,False,None)],
        "Periodical": [("id",1,9,False,None),("titles",2,9,True,None),("abstracts",20,9,True,None),
                       ("year",33,5,False,None),("doi",41,9,False,None)],
        "Thesis": [("id",1,9,False,None),("titles",3,9,True,None),("abstracts",18,9,True,None),
                   ("year",26,5,False,None),("doi",29,9,False,None)],
        "Conference": [("id",1,9,False,None),("titles",3,9,True,None),("abstracts",20,9,True,None),
                       ("year",28,5,False,None),("doi",32,9,False,None)],
        "Resource": [("type",1,9,False,None),("periodical",101,11,False,"Periodical"),
                     ("thesis",102,11,False,"Thesis"),("conference",104,11,False,"Conference")],
        "SearchResponse": [("status",1,8,False,None),("message",2,9,False,None),
                           ("count",3,3,False,None),("resources",4,11,True,"Resource")],
    }
    for name,fields in definitions.items():
        definition = schema.message_type.add(name=name)
        for field_name,number,kind,repeated,type_name in fields:
            field = definition.field.add(name=field_name,number=number,type=kind,label=3 if repeated else 1)
            if type_name:
                field.type_name = f".paperagent_wanfang.{type_name}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return {name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"paperagent_wanfang.{name}"))
            for name in definitions}


MESSAGES = _messages()


def encode_search(query, max_results):
    """按官网请求字段生成一个未压缩的 gRPC-Web 单消息帧。"""
    request = MESSAGES["SearchRequest"]()
    request.common.search_type = "paper"
    request.common.search_word = query
    request.common.current_page = 1
    request.common.page_size = max_results
    request.interface_type = 1
    payload = request.SerializeToString()
    return b"\x00" + struct.pack(">I",len(payload)) + payload


def decode_search(payload):
    """验证帧边界、服务状态与消息数量，然后返回标准 Protobuf 对象。"""
    messages = []
    position = 0
    while position < len(payload):
        if len(payload)-position < 5:
            raise ValueError("万方检索响应帧不完整")
        flag = payload[position]
        size = struct.unpack(">I",payload[position+1:position+5])[0]
        position += 5
        end = position + size
        if end > len(payload):
            raise ValueError("万方检索响应长度异常")
        body = payload[position:end]
        position = end
        if flag == 128:
            trailers = dict(line.split(":",1) for line in body.decode("ascii").split("\r\n") if ":" in line)
            if trailers.get("grpc-status","0").strip() != "0":
                raise RuntimeError("万方检索服务返回失败状态")
        elif flag == 0:
            messages.append(body)
        else:
            raise ValueError("万方检索返回了不支持的压缩帧")
    if len(messages) != 1:
        raise ValueError("万方检索需要返回一个结果消息")
    response = MESSAGES["SearchResponse"]()
    response.ParseFromString(messages[0])
    if not response.status:
        raise RuntimeError("万方检索失败：" + response.message[:200])
    return response
