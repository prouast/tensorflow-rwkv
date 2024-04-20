#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace rwkv {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("WKV")
    .Input("k: T") // (B, T, C)
    .Input("v: T") // (B, T, C)
    .Input("w: T") // -exp(time_decay) (C,)
    .Input("u: T") // time_first (C,)
    .Output("wkv: T") // (B, T, C)
    .Attr("T: {float, bfloat16, half}")
    .SetShapeFn([](InferenceContext* c) {
      // k and v
      ShapeHandle shp_hnd_k, shp_hnd_v, shp_hnd_kv;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &shp_hnd_k));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &shp_hnd_v));
      TF_RETURN_IF_ERROR(c->Merge(shp_hnd_k, shp_hnd_v, &shp_hnd_kv));
      int32 B, T, C;
      B = c->Value(c->Dim(shp_hnd_kv, 0));
      T = c->Value(c->Dim(shp_hnd_kv, 1));
      C = c->Value(c->Dim(shp_hnd_kv, 2));
      c->set_output(0, c->MakeShape({B, T, C}));
      // w and u
      ShapeHandle shp_hnd_w, shp_hnd_u;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shp_hnd_w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shp_hnd_u));
      return Status();
    })
    .Doc(R"Doc(WKV op.)Doc");

REGISTER_OP("WKVGrad")
    .Input("k: T") // (B, T, C)
    .Input("v: T") // (B, T, C)
    .Input("w: T") // -exp(time_decay) (C,)
    .Input("u: T") // time_first (C,)
    .Input("gwkv: T") // (B, T, C)
    .Output("gk: T") // (B, T, C)
    .Output("gv: T") // (B, T, C)
    .Output("gw: T") // (C,)
    .Output("gu: T") // (C,)
    .Attr("T: {float, bfloat16, half}")
    .SetShapeFn([](InferenceContext* c) {
      // k and v
      ShapeHandle shp_hnd_k, shp_hnd_v, shp_hnd_kv;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &shp_hnd_k));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &shp_hnd_v));
      TF_RETURN_IF_ERROR(c->Merge(shp_hnd_k, shp_hnd_v, &shp_hnd_kv));
      int32 B, T, C;
      B = c->Value(c->Dim(shp_hnd_kv, 0));
      T = c->Value(c->Dim(shp_hnd_kv, 1));
      C = c->Value(c->Dim(shp_hnd_kv, 2));
      c->set_output(0, c->MakeShape({B, T, C}));
      c->set_output(1, c->MakeShape({B, T, C}));
      // w and u
      ShapeHandle shp_hnd_w, shp_hnd_u, shp_hnd_wu;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shp_hnd_w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shp_hnd_u));
      TF_RETURN_IF_ERROR(c->Merge(shp_hnd_w, shp_hnd_u, &shp_hnd_wu));
      C = c->Value(c->Dim(shp_hnd_wu, 0));
      c->set_output(2, c->MakeShape({C}));
      c->set_output(3, c->MakeShape({C}));
      return Status();
    })
    .Doc(R"Doc(WKVGrad op.)Doc");

}  // namespace rwkv
}  // namespace tensorflow