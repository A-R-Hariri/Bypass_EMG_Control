import libemg
import numpy as np
import time
from datetime import datetime
import csv
import time
import threading
import sys
import socket 
import joblib
import os

from dynamixel_sdk import *
from dynamixel_sdk.protocol1_packet_handler import Protocol1PacketHandler

# ---- Monkey patch ----
old_bulkReadTx = Protocol1PacketHandler.bulkReadTx

def patched_bulkReadTx(self, port, param, param_length, *args):
    return old_bulkReadTx(self, port, param, param_length)

Protocol1PacketHandler.bulkReadTx = patched_bulkReadTx

COLLECT = 0
NAME = 'aibme'
PATH = 'pickles/'
MODEL = 1
dataset_folder = 'data/' + NAME + '/'
bypass_log = 'bypass_log/' + NAME + '/'


# ---- Control table address ----
RATE = 60

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_GOAL_VEL = 32
ADDR_TORQUE_LIMIT = 34 

ADDR_PRESENT_POSITION = 36
ADDR_PRESENT_VELOCITY = 38
ADDR_PRESENT_TORQUE = 40

PROTOCOL_VERSION = 1.0
WRIST_ID = 0 
GRIP_ID = 1 
BAUDRATE = 3000000  
DEVICENAME = 'COM6' 

MOVING_THRESHOLD = 30   
WRIST_MAX_TORQUE = 500
GRIP_MAX_TORQUE = 400
WRIST_MAX_VEL = 300
GRIP_MAX_VEL = 200
WRIST_MIN_POS = 800
WRIST_MAX_POS = 2500
GRIP_MAX_POS = 2340
GRIP_MIN_POS = 1780
GRIP_POS = [GRIP_MIN_POS, int(np.mean([GRIP_MIN_POS, GRIP_MAX_POS])), GRIP_MAX_POS]
WRIST_POS = [WRIST_MIN_POS, int(np.mean([WRIST_MIN_POS, WRIST_MAX_POS])), WRIST_MAX_POS]

# wrist_pos = 0
wrist_trq = WRIST_MAX_TORQUE
# grip_pos = 0
grip_trq = GRIP_MAX_TORQUE
# speed = 0
# speedG = 0
# speedW = 0

SharedContext = None

class Bypass():
    def __init__(self, shared_context):
        self.sc = shared_context

    def setup(self):
        self.port_handler = PortHandler(DEVICENAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        self.pos_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, 2)
        # trq_sync_write = GroupSyncWrite(port_handler, packet_handler, ADDR_TORQUE_LIMIT, 2)
        self.vel_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, ADDR_GOAL_VEL, 2)

        self.group_bulk_read = GroupBulkRead(self.port_handler, self.packet_handler)
        self.group_bulk_read.addParam(WRIST_ID, ADDR_PRESENT_POSITION, 6)
        # group_bulk_read.addParam(WRIST_ID, ADDR_PRESENT_VELOCITY, 2)
        # group_bulk_read.addParam(WRIST_ID, ADDR_PRESENT_TORQUE, 2)
        self.group_bulk_read.addParam(GRIP_ID, ADDR_PRESENT_POSITION, 6)
        # group_bulk_read.addParam(GRIP_ID, ADDR_PRESENT_VELOCITY, 2)
        # group_bulk_read.addParam(GRIP_ID, ADDR_PRESENT_TORQUE, 2)

        # ---- Open port ----
        if self.port_handler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            quit()

        # ---- Set port baudrate ----
        if self.port_handler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            quit()

        # ---- Enable Dynamixel Torque ----
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, WRIST_ID, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, WRIST_ID, ADDR_GOAL_VEL, WRIST_MAX_VEL)
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, WRIST_ID, ADDR_TORQUE_LIMIT, WRIST_MAX_TORQUE)

        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, GRIP_ID, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, GRIP_ID, ADDR_GOAL_VEL, GRIP_MAX_VEL)
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, GRIP_ID, ADDR_TORQUE_LIMIT, GRIP_MAX_TORQUE)


    def run(self):
        # ---- Logger ----
        if not os.path.exists(bypass_log):
            os.makedirs(bypass_log, exist_ok=True)
        log_file = open(bypass_log + f"bypass_log_{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}.csv", "w", newline="")
        logger = csv.writer(log_file)
        logger.writerow(["time", "w_pos_d", "w_vel_d", "w_trq_d", "g_pos_d", "g_vel_d", "g_trq_d",
                            "w_pos", "w_vel", "w_trq", "g_pos", "g_vel", "g_trq",
                            "velocity", "probs_0", "probs_1", "probs_2", "probs_3", "probs_4"])
        
        # ---- Control Loop ----
        interval = 1.0 / RATE
        next_time = time.perf_counter()

        while True:
            now = time.perf_counter()

            if now >= next_time:
                w_pos = self.sc.wrist_pos
                g_pos = self.sc.grip_pos
                w_trq = wrist_trq
                g_trq = grip_trq
                g_vel = int(GRIP_MAX_VEL * self.sc.speedG)
                w_vel = int(WRIST_MAX_VEL * self.sc.speedW)
                g_vel += 0 if g_vel else 1
                w_vel += 0 if w_vel else 1

                if w_pos not in [0, 1, 2] or g_pos not in [0, 1, 2]:
                    break

                # wrist_goal_trq = [DXL_LOBYTE(wrist_trq), DXL_HIBYTE(wrist_trq)]
                # grip_goal_trq = [DXL_LOBYTE(grip_trq), DXL_HIBYTE(grip_trq)]
                # wrist_addparam_result = trq_sync_write.addParam(WRIST_ID, wrist_goal_trq)
                # grip_addparam_result = trq_sync_write.addParam(GRIP_ID, grip_goal_trq)
                # dxl_comm_result = trq_sync_write.txPacket()
                # if dxl_comm_result != COMM_SUCCESS:
                #     print("%s" % packet_handler.getTxRxResult(dxl_comm_result))
                # trq_sync_write.clearParam()

                wrist_goal_pos = [DXL_LOBYTE(WRIST_POS[w_pos]), DXL_HIBYTE(WRIST_POS[w_pos])]
                grip_goal_pos = [DXL_LOBYTE(GRIP_POS[g_pos]), DXL_HIBYTE(GRIP_POS[g_pos])]
                wrist_addparam_result = self.pos_sync_write.addParam(WRIST_ID, wrist_goal_pos)
                grip_addparam_result = self.pos_sync_write.addParam(GRIP_ID, grip_goal_pos)
                dxl_comm_result = self.pos_sync_write.txPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
                self.pos_sync_write.clearParam()

                wrist_goal_vel = [DXL_LOBYTE(w_vel), DXL_HIBYTE(w_vel)]
                grip_goal_vel = [DXL_LOBYTE(g_vel), DXL_HIBYTE(g_vel)]
                wrist_addparam_result = self.vel_sync_write.addParam(WRIST_ID, wrist_goal_vel)
                grip_addparam_result = self.vel_sync_write.addParam(GRIP_ID, grip_goal_vel)
                dxl_comm_result = self.vel_sync_write.txPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
                self.vel_sync_write.clearParam()

                # ---- Feedback ----
                dxl_comm_result = self.group_bulk_read.txRxPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
                
                wrist_pres_pos = self.group_bulk_read.getData(WRIST_ID, ADDR_PRESENT_POSITION, 2)
                wrist_pres_vel = self.group_bulk_read.getData(WRIST_ID, ADDR_PRESENT_VELOCITY, 2)
                wrist_pres_trq = self.group_bulk_read.getData(WRIST_ID, ADDR_PRESENT_TORQUE, 2)

                grip_pres_pos = self.group_bulk_read.getData(GRIP_ID, ADDR_PRESENT_POSITION, 2)
                grip_pres_vel = self.group_bulk_read.getData(GRIP_ID, ADDR_PRESENT_VELOCITY, 2)
                grip_pres_trq = self.group_bulk_read.getData(GRIP_ID, ADDR_PRESENT_TORQUE, 2)

                logger.writerow([time.time(), w_pos, w_vel, w_trq, g_pos, g_vel, g_trq,
                                wrist_pres_pos, wrist_pres_vel, wrist_pres_trq,
                                grip_pres_pos, grip_pres_vel, grip_pres_trq,
                                self.sc.velocity, *self.sc.probs])

                next_time += interval
            else:
                time.sleep(next_time - now)

        print('Stopping')
        log_file.close()
        # ---- Disable Dynamixel Torque ----
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, WRIST_ID, ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))

        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, GRIP_ID, ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(dxl_error))

        # ---- Close port ----
        self.port_handler.closePort()



