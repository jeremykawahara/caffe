#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class WeightedHingeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WeightedHingeLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WeightedHingeLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WeightedHingeLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(WeightedHingeLossLayerTest, TestGradientMultiClassL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WeightedHingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WeightedHingeLossLayerTest, TestGradientMultiClassL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  WeightedHingeLossParameter* weighted_hinge_loss_param = layer_param.mutable_weighted_hinge_loss_param();
  
  weighted_hinge_loss_param->set_norm(WeightedHingeLossParameter_Norm_L2);
  WeightedHingeLossLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WeightedHingeLossLayerTest, TestGradientQuadraticL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  WeightedHingeLossParameter* weighted_hinge_loss_param = layer_param.mutable_weighted_hinge_loss_param();

  weighted_hinge_loss_param->set_weight_type(WeightedHingeLossParameter_WeightType_QUADRATIC);
  WeightedHingeLossLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WeightedHingeLossLayerTest, TestGradientQuadraticL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  WeightedHingeLossParameter* weighted_hinge_loss_param = layer_param.mutable_weighted_hinge_loss_param();

  weighted_hinge_loss_param->set_weight_type(WeightedHingeLossParameter_WeightType_QUADRATIC);
  weighted_hinge_loss_param->set_norm(WeightedHingeLossParameter_Norm_L2);

  WeightedHingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
