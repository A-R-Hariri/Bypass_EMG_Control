import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch, torch.nn as nn
import libemg
import numpy as np
import socket, threading, random, select
from datetime import datetime
from multiprocessing import Manager

from utils import * 
from models import CNN, CNN_GRL, MLP
from bypass import Bypass

SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class MultiModelWrapper(nn.Module):
    def __init__(self, models_dict, shared_context, 
                 feature_list=FEATURE_LIST, feature_dic=FEATURE_DIC,
                 device=DEVICE):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)
        self.device = device
        self.sc = shared_context
        self.active_name = None
        self.active_model = None
        self.feature_list = feature_list
        self.feature_dic = feature_dic
        self.fe = libemg.feature_extractor.FeatureExtractor()

    def forward(self, x):
        name = self.sc.active_model_name
        if name not in self.models:
            return None
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        
        name = self.sc.active_model_name
        if name != self.active_name:
            self.active_name = name
            self.active_model = self.models[name]
            print(f"[MODEL CHANGE]: {name}")
        return self.active_model(x)

    def predict_proba(self, x):
        if self.sc.active_model_name not in self.models: return np.zeros((1, 5))
        return self(x).detach().cpu().numpy()
        
    @torch.no_grad()
    def predict(self, x):
        if self.sc.active_model_name not in self.models: return np.zeros((1,))
        return self(x).detach().argmax(1).cpu().numpy()

def load_all_models(model_names):
    models = {}
    for name in model_names:
        if 'within' in name:
            w_path = join(SGT_PATH, f"{name}.pt")
        else:
            w_path = join(PATH, f"{name}.pt")
        if 'grl' in name:
            m = CNN_GRL().to(DEVICE)
        elif 'mlp' in name:
            m = MLP(48).to(DEVICE)
        else:
            m = CNN().to(DEVICE)
        m.load_state_dict(torch.load(w_path, map_location=DEVICE))
        m.eval()
        models[name] = m
        print(f"[SUCCESS] Loaded: {name}")
    return models

def input_thread(sockets_dict, sc):
    print("Input thread started...")
    socks_list = list(sockets_dict.values())
    while True:
        try:
            readable, _, _ = select.select(socks_list, [], [])
            for sock in readable:
                data, _ = sock.recvfrom(1024)
                
                name = sc.active_model_name
                if 'mlp' in name: active_cat = 'within_mlp'
                elif 'within' in name: active_cat = 'within_cnn'
                else: active_cat = 'normal'
                
                if sock != sockets_dict.get(active_cat):
                    continue

            parts = data.decode("utf-8").strip().split(' ')
            if len(parts) >= 6:
                probs_list = [float(p) for p in parts[:-2]]
                raw_vel = float(parts[-2])          # [-1] is timestamp
                gesture = np.argmax(np.array(probs_list))
                
                flip = sc.flip_lr
                speed = np.clip(raw_vel, 0.0, 1.0)

                if gesture == 1: command = [-1, 0]
                elif gesture == 4: command = [-1, 2]
                elif gesture == 2: command = [2, -1] if flip else [0, -1]
                elif gesture == 3: command = [0, -1] if flip else [2, -1]
                else: command = [-1, -1]

                w, g = int(command[0]), int(command[1])
                wrist_pos = w if w != -1 else sc.wrist_pos
                grip_pos = g if g != -1 else sc.grip_pos
                speedG = 0 if g == -1 else speed
                speedW = 0 if w == -1 else speed

                sc.speedG, sc.speedW = speedG, speedW
                sc.wrist_pos, sc.grip_pos = wrist_pos, grip_pos
                sc.probs, sc.velocity = probs_list, raw_vel
            
        except: pass

if __name__ == "__main__":
    manager = Manager()
    SharedContext = manager.Namespace()
    
    # Initialize Namespace values BEFORE spawning libemg
    SharedContext.speedG = 0.0
    SharedContext.speedW = 0.0
    SharedContext.wrist_pos = 0
    SharedContext.grip_pos = 0
    SharedContext.probs = [0.0]*5
    SharedContext.velocity = 0.0
    SharedContext.flip_lr = False
    SharedContext.speed_multiplier = 1.0
    
    SharedContext.params = PARAMS

    model_names = [
        'cnn_raw',
    ]
        
    loaded_models = load_all_models(model_names)
    SharedContext.available_models = list(loaded_models.keys())
    SharedContext.active_model_name = model_names[0]

    dict_normal = {k: v for k, v in loaded_models.items() if 'within' not in k and 'mlp' not in k}
    wrapper = MultiModelWrapper(dict_normal, SharedContext).to(DEVICE)
    o_class = libemg.emg_predictor.EMGClassifier(wrapper)

    o_class.add_velocity([], [])
    th_max_path = join(PATH, 'th_max_dic.npy')
    th_min_path = join(PATH, 'th_min_dic.npy')
    if os.path.exists(th_max_path):
        o_class.th_max_dic = np.load(th_max_path, allow_pickle=True).item()
        o_class.th_min_dic = np.load(th_min_path, allow_pickle=True).item()

    p, smm = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smm)
    p = 12346
    on = libemg.emg_predictor.OnlineEMGClassifier(o_class, SEQ, INC, odh, 
                                                        ip='127.0.0.1', port=p,
                                                        features=None, output_format='probabilities')
    
    on.run(block=False)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('127.0.0.1', p))
    sockets_dict = {'normal': s}

    threading.Thread(target=input_thread, args=(sockets_dict, SharedContext), daemon=True).start()

    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH, exist_ok=True)
    odh.log_to_file(file_path=join(DATA_PATH, f"bypass_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))

    bp = Bypass(shared_context=SharedContext)
    bp.setup()
    bp.run()
    print('done')