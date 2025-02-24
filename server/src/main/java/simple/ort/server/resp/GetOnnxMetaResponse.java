package simple.ort.server.resp;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxModelMetadata;

import java.util.Map;

public record GetOnnxMetaResponse(Map<String, NodeInfo> inputMeta, Map<String, NodeInfo> outputMeta,
                                  OnnxModelMetadata modelMeta) {
}
