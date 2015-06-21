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
    for (int j = 0; j < dim; ++j) {
      // Label-wise ranking weight
      Dtype weight = std::exp( std::pow(static_cast<int>(label[i]) - j, 2) / std::pow(dim - 1, 2) );

      // Label-wise margin for each
      Dtype margin = ( j == static_cast<int>(label[i]) ) ? 
          ( 1.0 + bottom_diff[i * dim + j] ): 
          ( 1.0 - bottom_diff[i * dim + j] );

      switch (this->layer_param_.weighted_hinge_loss_param().norm()) {
        case WeightedHingeLossParameter_Norm_L1:{
          bottom_diff[i * dim + j] = std::max(Dtype(0), margin) * weight;
          break;
        }
        case WeightedHingeLossParameter_Norm_L2:{
          bottom_diff[i * dim + j] = std::pow(std::max(Dtype(0), margin),2) * weight;
          break;
        }
        default:
          LOG(FATAL) << "Unknown Norm";
      }
    }
  }

  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.weighted_hinge_loss_param().norm()) {
    case WeightedHingeLossParameter_Norm_L1:{
      loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
      break;
    }
    case WeightedHingeLossParameter_Norm_L2:{
      loss[0] = caffe_cpu_asum(count, bottom_diff) / num; 
      break;
    }
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
      for (int j = 0; j < dim; ++j) 
      {
        //==============================================================
        // Label-wise ranking weight
        Dtype weight = std::exp( std::pow(static_cast<int>(label[i]) - j, 2) / std::pow(dim - 1, 2) );

        //==============================================================
        // Label-wise margin for each
        Dtype margin = ( j == static_cast<int>(label[i]) )? 
          ( 1.0 + bottom_diff[i * dim + j] ):
          ( 1.0 - bottom_diff[i * dim + j] );

        switch (this->layer_param_.weighted_hinge_loss_param().norm()) {
          case WeightedHingeLossParameter_Norm_L1:{
            if( margin > Dtype(0) )
            {
              bottom_diff[i * dim + j] = ( j == static_cast<int>(label[i]) )? 
                (  weight): 
                (- weight);
            }
            else
              bottom_diff[i * dim + j] = Dtype(0);          
            break;
          }
          case WeightedHingeLossParameter_Norm_L2:
          {
            if( margin > Dtype(0) )
            {
              bottom_diff[i * dim + j] = ( j == static_cast<int>(label[i]) )? 
              (  2 * weight * margin): 
              (- 2 * weight * margin);
            }
            else
              bottom_diff[i * dim + j] = Dtype(0);
            break;
          }
          default:
            LOG(FATAL) << "Unknown Norm";
        }

      } // end for (int j = 0; j < dim; ++j) 
    } // end for (int i = 0; i < num; ++i) 

    // Finally normalize the backwarded gradient 
    switch (this->layer_param_.weighted_hinge_loss_param().norm()) {
      case WeightedHingeLossParameter_Norm_L1:
      {
        const Dtype loss_weight = top[0]->cpu_diff()[0];
        caffe_scal(count, loss_weight / num, bottom_diff);
        break;
      }
      case WeightedHingeLossParameter_Norm_L2:
      {
        const Dtype loss_weight = top[0]->cpu_diff()[0];
        caffe_scal(count, loss_weight *2 / num, bottom_diff);
        break;
      }
    }
  }
}

INSTANTIATE_CLASS(WeightedHingeLossLayer);
REGISTER_LAYER_CLASS(WeightedHingeLoss);

} // namespace caffe
