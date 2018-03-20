function [ output_args ] = cIRM_Net_Change( input_args )
%CIRM_NET_CHANGE Summary of this function goes here
%   Detailed explanation goes here


    % File_Data = 'DNN_datas.mat';
    % File_Param = 'DNN_params.mat';
    % File_NET = 'DNN_net.mat';
    % File_TRAIN_NET = 'DNN_CIRM_net.mat';

    File_Data = 'DNN_datas.mat';
    File_Param = 'DNN_params.mat';

    File_NET = 'DNN_net_03.mat';
    File_TRAIN_NET = 'DNN_CIRM_net_03.mat';

    %% Change Net Struct
    load(File_NET);
    [r,c] = size(struct_net);

    % new resized NET Struct
    net=[];

    for x = (1:c)
        dummy_struct.W = struct_net(x).W;
        dummy_struct.b = struct_net(x).b;
        dummy_struct.Wo1 = struct_net(x).Wo1;
        dummy_struct.bo1 = struct_net(x).bo1;
        dummy_struct.Wo2 = struct_net(x).Wo2;
        dummy_struct.bo2 = struct_net(x).bo2;

        if (x ==1)
            net = dummy_struct;
        else
            net = [net;dummy_struct];
        end

        fprintf('*** NET: %d\t', size(net) );
        fprintf('\n');

        fprintf('W: %d\t', size(dummy_struct.W) );
        fprintf('\n');
        fprintf('b: %d\t', size(dummy_struct.b) );
        fprintf('\n');
        fprintf('Wo1: %d\t', size(dummy_struct.Wo1) );
        fprintf('\n');
        fprintf('bo1: %d\t', size(dummy_struct.bo1) );
        fprintf('\n');
        fprintf('Wo2: %d\t', size(dummy_struct.Wo2) );
        fprintf('\n');
        fprintf('bo2: %d\t', size(dummy_struct.bo2) );
        fprintf('\n');

    end
    %%
    % Opts param here
    load(File_Param)
    %%
    save(File_TRAIN_NET,'net','opts','-v7.3');
    %%
end

