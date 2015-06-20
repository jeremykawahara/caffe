#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  // Copy bottom activation to bottom differentiation
  caffe_copy(count, bottom_data, bottom_diff);

  for (int i = 0; i < num; ++i) {
    //==============================================================
    // Cache corect_activation for this sample
    Dtype correct_activation = bottom_diff[i * dim + static_cast<int>(label[i])];
    bottom_diff[i * dim + static_cast<int>(label[i])] = 0;
    
    for (int j = 0; j < dim; ++j) {
      if( j == static_cast<int>(label[i]) ) continue;
      // Pairwise ranking weight

      Dtype weight = 1;
      switch (this->layer_param_.weighted_hinge_loss_param().weight_type()) {
        case WeightedHingeLossParameter_WeightType_MULTICLASS:
          weight = 1;
          break;
        case WeightedHingeLossParameter_WeightType_QUADRATIC:
          weight = std::pow(static_cast<int>(label[i]) - j, 2) / std::pow(dim - 1, 2);
          break;
        default:
          LOG(FATAL) << "Unknown Weight";
      }
      
      // Pairwise margin for each 
      Dtype margin = 1 + bottom_diff[i * dim + j] - correct_activation;

      bottom_diff[i * dim + j] = std::max(Dtype(0), margin) * weight;
    }
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.weighted_hinge_loss_param().norm()) {
  case WeightedHingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case WeightedHingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void WeightedHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    // Copy bottom activation to bottom differentiation
    caffe_copy(count, bottom_data, bottom_diff);

    for (int i = 0; i < num; ++i) {
      //==============================================================
      // Cache corect_activation for this sample
      Dtype correct_activation = bottom_diff[i * dim + static_cast<int>(label[i])];
      bottom_diff[i * dim + static_cast<int>(label[i])] = 0;

      for (int j = 0; j < dim; ++j) {
        if( j == static_cast<int>(label[i]) ) continue;
        //==============================================================
        // Pairwise ranking weight
        Dtype weight = 1;
        switch (this->layer_param_.weighted_hinge_loss_param().weight_type()) {
          case WeightedHingeLossParameter_WeightType_MULTICLASS:
            weight = 1;
            break;
          case WeightedHingeLossParameter_WeightType_QUADRATIC:
            weight = std::pow(static_cast<int>(label[i]) - j, 2) / std::pow(dim - 1, 2);
            break;
          default:
            LOG(FATAL) << "Unknown Weight";
        }
	      // LOG(INFO) << "weight on back propagate phase: " << weight << "\n";

        //==============================================================
        // Pairwise margin for each
        Dtype margin = 1 + bottom_diff[i * dim + j] - correct_activation;

        const Dtype loss_weight = top[0]->cpu_diff()[0];
        switch (this->layer_param_.weighted_hinge_loss_param().norm()) {
          case WeightedHingeLossParameter_Norm_L1:

            if( margin > Dtype(0) )
            {
            // LOG(INFO) << "bottom_diff[" << (i * dim + j) << "] = "<< weight << "\n";
            // LOG(INFO) << "bottom_diff[" << (i * dim + static_cast<int>(label[i])) << "] = " 
            //  << ( bottom_diff[i * dim +  static_cast<int>(label[i])] - weight) << "\n";
              bottom_diff[i * dim + j] = weight;
              bottom_diff[i * dim +  static_cast<int>(label[i])] -= weight;
            }
            else
              bottom_diff[i * dim + j] = Dtype(0);
          
            caffe_scal(count, loss_weight / num, bottom_diff);
          
            break;
          case WeightedHingeLossParameter_Norm_L2:

            if( margin > Dtype(0) )
            {
            // LOG(INFO) << "bottom_diff[" << (i * dim + j) << "] = "<< weight << "\n";
            // LOG(INFO) << "bottom_diff[" << (i * dim + static_cast<int>(label[i])) << "] = " 
            //  << ( bottom_diff[i * dim +  static_cast<int>(label[i])] - weight) << "\n";
              bottom_diff[i * dim + j] *= weight;
              bottom_diff[i * dim +  static_cast<int>(label[i])] -= weight*correct_activation;
            }
            else
              bottom_diff[i * dim + j] = Dtype(0);

            caffe_scal(count, loss_weight * 2 / num, bottom_diff);
            break;
          default:
            LOG(FATAL) << "Unknown Norm";
        }

      }
    }

  }
}

INSTANTIATE_CLASS(WeightedHingeLossLayer);
REGISTER_LAYER_CLASS(WeightedHingeLoss);

} // namespace caffe
