#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace rwkv {
namespace functor {

REGISTER_OP("WKV")
    .Input("k: T") // (B, N, C)
    .Input("v: T") // (B, N, C)
    .Input("w: T") // (C,)
    .Input("u: T") // (C,)
    .Output("wkv: T") // (B, N, C)
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      // k and v
      ShapeHandle shp_hnd_k, shp_hnd_v, shp_hnd_kv;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &shp_hnd_k));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &shp_hnd_v));
      TF_RETURN_IF_ERROR(c->Merge(shp_hnd_k, shp_hnd_v, &shp_hnd_kv));
      int32 B, N, C;
      B = c->Value(c->Dim(shp_hnd_kv, 0));
      N = c->Value(c->Dim(shp_hnd_kv, 1));
      C = c->Value(c->Dim(shp_hnd_kv, 2));
      c->set_output(0, c->MakeShape({B, N, C}));
      // w and u
      ShapeHandle shp_hnd_w, shp_hnd_u;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shp_hnd_w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shp_hnd_u));
      return Status();
    })
    .Doc(R"Doc(WKV op.)Doc");

REGISTER_OP("WKVGrad")
    .Input("k: T") // (B, N, C)
    .Input("v: T") // (B, N, C)
    .Input("w: T") // (C,)
    .Input("u: T") // (C,)
    .Input("gwkv: T") // (B, N, C)
    .Output("gk: T") // (B, N, C)
    .Output("gv: T") // (B, N, C)
    .Output("gw: T") // (B, C)
    .Output("gu: T") // (B, C)
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      // k and v
      ShapeHandle shp_hnd_k, shp_hnd_v, shp_hnd_kv;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &shp_hnd_k));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &shp_hnd_v));
      TF_RETURN_IF_ERROR(c->Merge(shp_hnd_k, shp_hnd_v, &shp_hnd_kv));
      int32 B, N, C;
      B = c->Value(c->Dim(shp_hnd_kv, 0));
      N = c->Value(c->Dim(shp_hnd_kv, 1));
      C = c->Value(c->Dim(shp_hnd_kv, 2));
      c->set_output(0, c->MakeShape({B, N, C}));
      c->set_output(1, c->MakeShape({B, N, C}));
      // w and u
      ShapeHandle shp_hnd_w, shp_hnd_u, shp_hnd_wu;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shp_hnd_w));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shp_hnd_u));
      TF_RETURN_IF_ERROR(c->Merge(shp_hnd_w, shp_hnd_u, &shp_hnd_wu));
      C = c->Value(c->Dim(shp_hnd_wu, 0));
      c->set_output(2, c->MakeShape({B, C}));
      c->set_output(3, c->MakeShape({B, C}));
      return Status();
    })
    .Doc(R"Doc(WKVGrad op.)Doc");

}  // namespace functor
}  // namespace rwkv
}  // namespace tensorflow