class Component_Specs:

    def __init__(self):

        self.LR_Mapping = {1: 0.0001,
                            2: 0.001,
                            3: 0.01,
                            4: 0.1}

        self.OPT_Mapping = {1: "RMSProp",
                          2: "SGD",
                          3: "Adam",
                          4: "Adagrad"}

        self.COMPONENT_Mapping = {1: self._FCNN_64x64_ReLu(),
                                  2: self._FCNN_64x64_Sigmoid(),
                                  3: self._Cnv1D_64_64_3(),
                                  4: self._Cnv2D_64_64_3()}

        self.DROPOUT_P_Mapping = {1: 0.01,
                                2: 0.1,
                                3: 0.2,
                                4: 0.3,
                                5: 0.4,
                                6: 0.5,
                                7: 0.6,
                                8: 0.7,
                                9: 0.8,
                                10: 0.9}
        self.specs = {"LR_Mapping":  self.LR_Mapping,
                    "OPT_Mapping":  self.OPT_Mapping,
                    "COMPONENT_Mapping":  self.COMPONENT_Mapping,
                    "DROPOUT_P_Mapping":  self.DROPOUT_P_Mapping}

    def return_component_mappings(self):
        return self.specs

    def _FCNN_64x64_ReLu(self):
        model_structure =  {
                             "0":{"layer_size_mapping": {"in_features": 64,
                                                         "out_features": 64},
                                 "layer_type": "linear",
                                 "activation": "relu"}
                            }
        return model_structure

    def _FCNN_64x64_Sigmoid(self):
        model_structure =  {
                             "0":{"layer_size_mapping": {"in_features": 64,
                                                         "out_features": 64},
                                 "layer_type": "linear",
                                 "activation": "sigmoid"}
                            }
        return model_structure

    def _Cnv1D_64_64_3(self):
        model_structure =  {
                             "0":{"layer_size_mapping": {"in_channels": 64,
                                                         "out_channels": 64,
                                                         "kernel_size": 3},
                                 "layer_type": "conv1d",
                                 "activation": "relu"}
                            }
        return model_structure

    def _Cnv2D_64_64_3(self):
        model_structure =  {
                             "0":{"layer_size_mapping": {"in_channels": 64,
                                                         "out_channels": 64,
                                                         "kernel_size": 3},
                                 "layer_type": "conv2d",
                                 "activation": "relu"}
                            }
        return model_structure
