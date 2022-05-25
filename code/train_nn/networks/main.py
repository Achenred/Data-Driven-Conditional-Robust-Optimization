from .mine_sp import MINE_SP
from .mine_shallow import MINE_SHALLOW
from .mine_gen import MINE_GEN
from .port_gen import PORT_GEN
from .mine_soft_assign import soft_assign,main_net
from .mine_soft_assign_AE1 import mine_Encoder1,mine_Decoder1,mine_Soft_KMeansCriterion1,mine_main_net_AE1
from .mine_soft_assign_AE2 import mine_Encoder2,mine_Decoder2,mine_Soft_KMeansCriterion2,mine_main_net_AE2
from .mine_soft_assign_AE3 import mine_Encoder3,mine_Decoder3,mine_Soft_KMeansCriterion3,mine_main_net_AE3

from .portfolio_soft_assign import conditional_assign,main_network


def build_network(net_name,main_size,encode_input_size=1,out_size=1,beta=1, lmbda=1,n_class=1):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 
                            'cifar10_LeNet_ELU','mine_sp','mine_shallow',
                            'mine_gen','port_gen','soft_assign','port_soft_assign',
                            'soft_assign_AE','soft_assign_AE1','soft_assign_AE2',
                            'soft_assign_AE3','port_soft_assign_AE1','port_soft_assign_AE2',
                            'port_soft_assign_AE3','deep_kmeans_AE')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mine_sp':
        net = MINE_SP()
        
    if net_name == 'mine_shallow':
        net = MINE_SHALLOW()
        
    if net_name == 'mine_gen':
        net = MINE_GEN()
        
    if net_name == 'port_gen':
        net = PORT_GEN(main_size,out_size)

    if net_name == 'soft_assign':
        return soft_assign(),main_net(n_class)
    
    if net_name == 'soft_assign_AE1':
        return mine_Encoder1(encode_input_size,out_size),mine_Decoder1(encode_input_size,out_size),mine_Soft_KMeansCriterion1(beta, lmbda),mine_main_net_AE1(n_class)
    
    if net_name == 'soft_assign_AE2':
        return mine_Encoder2(encode_input_size,out_size),mine_Decoder2(encode_input_size,out_size),mine_Soft_KMeansCriterion2(beta, lmbda),mine_main_net_AE2(n_class)
    
    if net_name == 'soft_assign_AE3':
        return mine_Encoder3(encode_input_size,out_size),mine_Decoder3(encode_input_size,out_size),mine_Soft_KMeansCriterion3(beta, lmbda),mine_main_net_AE3(n_class)
    
    if net_name == 'port_soft_assign':
        return conditional_assign(),main_network(n_class)
    
    if net_name == 'port_soft_assign_AE1':
        from .port_soft_assign_AE1 import Encoder1,Decoder1,Soft_KMeansCriterion1,main_net_AE1
        return Encoder1(encode_input_size,out_size),Decoder1(encode_input_size,out_size),Soft_KMeansCriterion1(beta, lmbda),main_net_AE1(n_class,main_size,out_size)
    
    if net_name == 'port_soft_assign_AE2':
        from .port_soft_assign_AE2 import Encoder2,Decoder2,Soft_KMeansCriterion2,main_net_AE2
        return Encoder2(encode_input_size,out_size),Decoder2(encode_input_size,out_size),Soft_KMeansCriterion2(beta, lmbda),main_net_AE2(n_class,main_size,out_size)
    
    if net_name == 'port_soft_assign_AE3':
        from .port_soft_assign_AE3 import Encoder3,Decoder3,Soft_KMeansCriterion3,main_net_AE3
        return Encoder3(encode_input_size,out_size),Decoder3(encode_input_size,out_size),Soft_KMeansCriterion3(beta, lmbda),main_net_AE3(n_class,main_size,out_size)
    
    if net_name == 'deep_kmeans_AE':
        from .mine_deep_kmeans_AE import Encoder,Decoder,KMeansCriterion,main_net_AE
        return Encoder(encode_input_size,out_size),Decoder(encode_input_size,out_size),KMeansCriterion(lmbda),main_net_AE()
    
    return net

