import os
import argparse
from dataclasses import dataclass, asdict

# =================================================================================================
# CONSTANTS & CONFIGURATION
# =================================================================================================

MEM_SIZE = 1000  # Arbitrary small size for assignment purposes
WORD_SIZE = 4
REG_COUNT = 32
INSTR_SIZE = 32

OPCODE_R_TYPE = 0b0110011
OPCODE_I_TYPE = 0b0010011
OPCODE_J_TYPE = 0b1101111
OPCODE_B_TYPE = 0b1100011
OPCODE_LW     = 0b0000011
OPCODE_SW     = 0b0100011

OPCODE_STR_R_TYPE = "0110011"
OPCODE_STR_I_TYPE_1 = "0010011"
OPCODE_STR_I_TYPE_2 = "0000011" # LW
OPCODE_STR_J_TYPE = "1101111"
OPCODE_STR_B_TYPE = "1100011"
OPCODE_STR_S_TYPE = "0100011"

# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================

def int_to_bin(value: int, n_bits: int = 32) -> str:
    """Convert integer to binary string with specified bit width."""
    binary = bin(value & (2**n_bits - 1))[2:]
    return "0" * (n_bits - len(binary)) + binary

def bin_to_int(binary: str, sign_ext: bool = False) -> int:
    """Convert binary string to integer with optional sign extension."""
    binary = str(binary)
    if sign_ext and binary[0] == "1":
        return -(-int(binary, 2) & (2 ** len(binary) - 1))
    return int(binary, 2)

def sign_extend(value: int, sign_bit: int) -> int:
    """Sign extend a value based on the sign bit position."""
    if (value & (1 << sign_bit)) != 0:
        value = value - (1 << (sign_bit + 1))
    return value

# =================================================================================================
# MEMORY & REGISTER COMPONENTS
# =================================================================================================

class InsMem:
    """Instruction memory for reading program instructions."""

    def __init__(self, name: str, io_dir: str):
        self.id = name
        with open(os.path.join(io_dir, "imem.txt"), "r") as f:
            self.mem = [line.strip() for line in f.readlines()]

    def read_instruction_hex(self, address: int) -> str:
        """Read instruction and return as hex string."""
        binary_str = "".join(self.mem[address:address + WORD_SIZE])
        if not binary_str: return "00000000" # Handle potential empty read
        instr = int(binary_str, 2)
        return format(instr, '#010x')

    def read_instruction_bin(self, address: int) -> str:
        """Read instruction and return as binary string."""
        return "".join(self.mem[address:address + WORD_SIZE])
    
class DataMem:
    """Data memory for load/store operations."""

    def __init__(self, name: str, io_dir: str):
        self.id = name
        self.io_dir = io_dir
        with open(os.path.join(io_dir, "dmem.txt"), "r") as f:
            self.mem = [line.strip() for line in f.readlines()]
        # Pad memory
        self.mem.extend(['00000000'] * (MEM_SIZE - len(self.mem)))

    def read_data_hex(self, addr: int) -> str:
        """Read data returning hex (Single Stage)."""
        binary_str = "".join(self.mem[addr : addr + WORD_SIZE])
        return format(int(binary_str, 2), '#010x')

    def write_data_hex(self, addr: int, write_data: int):
        """Write integer data into memory (Single Stage)."""
        mask8 = 0xFF
        for i in range(WORD_SIZE):
            byte_val = (write_data >> (8 * (3 - i))) & mask8 if False else (write_data >> (8 * i)) & mask8 
            self.mem[addr + i] = format((write_data >> (8 * (3-i))) & mask8, '08b')

    def write_data_ss_logic(self, addr: int, write_data: int):
        """Strict logic copy of original single stage write."""
        mask8 = 0xFF
        data8_arr = []
        temp_data = write_data
        for _ in range(WORD_SIZE):
            data8_arr.append(temp_data & mask8)
            temp_data = temp_data >> 8
        
        for i in range(WORD_SIZE):
            self.mem[addr + i] = format(data8_arr.pop(), '08b')

    def read_data_bin(self, addr_str: str) -> str:
        """Read 32 bits based on binary address string (Five Stage)."""
        addr_int = bin_to_int(addr_str)
        return "".join(self.mem[addr_int : addr_int + WORD_SIZE])

    def write_data_bin(self, addr_str: str, write_data_str: str):
        """Write 32 bit string to memory (Five Stage)."""
        addr_int = bin_to_int(addr_str)
        for i in range(WORD_SIZE):
            self.mem[addr_int + i] = write_data_str[8 * i : 8 * (i + 1)]

    def output_data_mem(self):
        res_path = os.path.join(self.io_dir, f"{self.id}_DMEMResult.txt")
        with open(res_path, "w") as f:
            f.writelines([f"{data}\n" for data in self.mem])

class RegisterFile:
    """Register file for processor registers."""

    def __init__(self, io_dir: str, prefix: str = ""):
        #prefix to differentiate output files (SS_RFResult vs FS_RFResult)
        filename = f"{prefix}_RFResult.txt" if prefix else "RFResult.txt"
        self.output_file = os.path.join(io_dir, filename)
        
        self.registers_int = [0x0] * REG_COUNT  # For Single Cycle
        self.registers_bin = [int_to_bin(0)] * REG_COUNT # For Five Stage

    # Single Cycle Methods
    def read_rf_int(self, reg_addr: int) -> int:
        return self.registers_int[reg_addr]

    def write_rf_int(self, reg_addr: int, data: int):
        if reg_addr != 0:
            self.registers_int[reg_addr] = data & 0xFFFFFFFF

    def output_rf_int(self, cycle: int):
        lines = [f"State of RF after executing cycle:  {cycle}\n"]
        lines.extend([format(val, '032b') + "\n" for val in self.registers_int])
        mode = "w" if cycle == 0 else "a"
        with open(self.output_file, mode) as f:
            f.writelines(lines)

    # Five Stage Methods
    def read_rf_str(self, reg_addr_bin: str) -> str:
        return self.registers_bin[bin_to_int(reg_addr_bin)]

    def write_rf_str(self, reg_addr_bin: str, data_str: str):
        if reg_addr_bin == "00000": return
        self.registers_bin[bin_to_int(reg_addr_bin)] = data_str

    def output_rf_str(self, cycle: int):
        """Output register state for five-stage."""
        lines = ["-" * 70 + "\n", 
                 f"State of RF after executing cycle:{cycle}\n"]
        lines.extend([f"{val}\n" for val in self.registers_bin])
        mode = "w" if cycle == 0 else "a"
        with open(self.output_file, mode) as f:
            f.writelines(lines)

# =================================================================================================
# Single-Stage Processor
# =================================================================================================

class StateSS:
    def __init__(self):
        self.IF = {"nop": False, "PC": 0, "taken": False}

class SingleStageCore:
    def __init__(self, io_dir: str, imem: InsMem, dmem: DataMem):
        self.io_dir = io_dir
        self.op_file_path = os.path.join(io_dir, "StateResult_SS.txt")
        self.imem = imem
        self.dmem = dmem
        self.rf = RegisterFile(io_dir, prefix="SS")
        self.state = StateSS()
        self.next_state = StateSS()
        self.cycle = 0
        self.inst_count = 0
        self.halted = False

    def step(self):
        # Fetch
        pc = self.state.IF["PC"]
        fetched_instr_hex = self.imem.read_instruction_hex(pc)
        fetched_instr_int = int(fetched_instr_hex, 16)
        
        # Decode & Execute
        self.decode_and_execute(fetched_instr_int)

        # Check Halt
        self.halted = self.state.IF["nop"]
        
        # Update PC
        if not self.state.IF["taken"] and (pc + 4 < len(self.imem.mem)):
            self.next_state.IF["PC"] = pc + 4
        else:
            self.state.IF["taken"] = False # Reset taken flag

        # Output
        self.rf.output_rf_int(self.cycle)
        self.print_state(self.next_state, self.cycle)

        # Update State
        self.state = self.next_state
        self.cycle += 1
        self.inst_count += 1

    def decode_and_execute(self, instr: int):
        """Decode and execute instruction based on opcode."""
        opcode = instr & 0x7F
        
        # R-Type
        if opcode == OPCODE_R_TYPE:
            funct7 = instr >> 25
            funct3 = (instr >> 12) & 0x7
            rs2 = (instr >> 20) & 0x1F
            rs1 = (instr >> 15) & 0x1F
            rd = (instr >> 7) & 0x1F
            
            val1 = self.rf.read_rf_int(rs1)
            val2 = self.rf.read_rf_int(rs2)
            
            res = 0
            if funct7 == 0 and funct3 == 0b000: res = val1 + val2       # ADD
            elif funct7 == 0b0100000 and funct3 == 0b000: res = val1 - val2 # SUB
            elif funct7 == 0 and funct3 == 0b100: res = val1 ^ val2     # XOR
            elif funct7 == 0 and funct3 == 0b110: res = val1 | val2     # OR
            elif funct7 == 0 and funct3 == 0b111: res = val1 & val2     # AND
            
            self.rf.write_rf_int(rd, res)

        # I-Type
        elif opcode == OPCODE_I_TYPE:
            imm = (instr >> 20) & 0xFFF
            funct3 = (instr >> 12) & 0x7
            rs1 = (instr >> 15) & 0x1F
            rd = (instr >> 7) & 0x1F
            
            val1 = self.rf.read_rf_int(rs1)
            imm_sext = sign_extend(imm, 11)
            
            res = 0
            if funct3 == 0b000: res = val1 + imm_sext   # ADDI
            elif funct3 == 0b100: res = val1 ^ imm_sext # XORI
            elif funct3 == 0b110: res = val1 | imm_sext # ORI
            elif funct3 == 0b111: res = val1 & imm_sext # ANDI
            
            self.rf.write_rf_int(rd, res)

        # J-Type (JAL)
        elif opcode == OPCODE_J_TYPE:
            # Reconstruct Immediate
            imm19_12 = (instr >> 12) & 0xFF
            imm11 = (instr >> 20) & 1
            imm10_1 = (instr >> 21) & 0x3FF
            imm20 = (instr >> 31) & 1
            imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1)
            
            rd = (instr >> 7) & 0x1F
            
            self.rf.write_rf_int(rd, self.state.IF["PC"] + 4)
            self.next_state.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 20)
            self.state.IF["taken"] = True

        # B-Type
        elif opcode == OPCODE_B_TYPE:
            imm11 = (instr >> 7) & 1
            imm4_1 = (instr >> 8) & 0xF
            imm10_5 = (instr >> 25) & 0x3F
            imm12 = (instr >> 31) & 1
            imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1)
            
            rs2 = (instr >> 20) & 0x1F
            rs1 = (instr >> 15) & 0x1F
            funct3 = (instr >> 12) & 0x7
            
            val1 = self.rf.read_rf_int(rs1)
            val2 = self.rf.read_rf_int(rs2)
            
            take = False
            if funct3 == 0b000: take = (val1 == val2) # BEQ
            else: take = (val1 != val2)               # BNE
            
            if take:
                self.next_state.IF["PC"] = self.state.IF["PC"] + sign_extend(imm, 12)
                self.state.IF["taken"] = True

        # LW
        elif opcode == OPCODE_LW:
            imm = instr >> 20
            rs1 = (instr >> 15) & 0x1F
            rd = (instr >> 7) & 0x1F
            
            addr = self.rf.read_rf_int(rs1) + sign_extend(imm, 11)
            mem_val = int(self.dmem.read_data_hex(addr), 16)
            self.rf.write_rf_int(rd, mem_val)

        # SW
        elif opcode == OPCODE_SW:
            imm11_5 = instr >> 25
            imm4_0 = (instr >> 7) & 0x1F
            imm = (imm11_5 << 5) | imm4_0
            rs1 = (instr >> 15) & 0x1F
            rs2 = (instr >> 20) & 0x1F
            
            addr = (self.rf.read_rf_int(rs1) + sign_extend(imm, 11)) & 0xFFFFFFFF
            self.dmem.write_data_ss_logic(addr, self.rf.read_rf_int(rs2))

        # HALT
        else:
            self.state.IF["nop"] = True

    def print_state(self, state, cycle):
        lines = ["-" * 70 + "\n",
            f"State after executing cycle: {cycle}\n",
            f"IF.PC: {state.IF['PC']}\n",
            f"IF.nop: {state.IF['nop']}\n",
        ]
        mode = "w" if cycle == 0 else "a"
        with open(self.op_file_path, mode) as f:
            f.writelines(lines)

# =================================================================================================
# Five-Stage Processor Pipeline States
# =================================================================================================

@dataclass
class IFState:
    """Instruction Fetch stage state."""
    nop: bool = False
    PC: int = 0

@dataclass
class IDState:
    """Instruction Decode stage state."""
    nop: bool = True
    hazard_nop: bool = False
    PC: int = 0
    instr: str = "0" * 32
    
    def get_dict_copy(self):
        return {"nop": self.nop, "Instr": self.instr[::-1]}

@dataclass
class EXState:
    """Execute stage state."""
    nop: bool = True
    instr: str = ""
    read_data_1: str = "0" * 32
    read_data_2: str = "0" * 32
    imm: str = "0" * 32
    rs: str = "0" * 5
    rt: str = "0" * 5
    write_reg_addr: str = "0" * 5
    is_I_type: bool = False
    read_mem: bool = False
    write_mem: bool = False
    alu_op: str = "00"
    write_enable: bool = False

    def get_dict_copy(self):
        return {
            "nop": self.nop,
            "instr": self.instr[::-1],
            "Read_data1": self.read_data_1,
            "Read_data2": self.read_data_2,
            "Imm": self.imm,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "is_I_type": int(self.is_I_type),
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "alu_op": self.alu_op,
            "wrt_enable": int(self.write_enable),
        }

@dataclass
class MEMState:
    """Memory Access stage state."""
    nop: bool = True
    alu_result: str = "0" * 32
    store_data: str = "0" * 32
    rs: str = "0" * 5
    rt: str = "0" * 5
    write_reg_addr: str = "0" * 5
    read_mem: bool = False
    write_mem: bool = False
    write_enable: bool = False

    def get_dict_copy(self):
        return {
            "nop": self.nop,
            "ALUresult": self.alu_result,
            "Store_data": self.store_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "rd_mem": int(self.read_mem),
            "wrt_mem": int(self.write_mem),
            "wrt_enable": int(self.write_enable),
        }

@dataclass
class WBState:
    """Write Back stage state."""
    nop: bool = True
    write_data: str = "0" * 32
    rs: str = "0" * 5
    rt: str = "0" * 5
    write_reg_addr: str = "0" * 5
    write_enable: bool = False

    def get_dict_copy(self):
        return {
            "nop": self.nop,
            "Wrt_data": self.write_data,
            "Rs": self.rs,
            "Rt": self.rt,
            "Wrt_reg_addr": self.write_reg_addr,
            "wrt_enable": int(self.write_enable),
        }

class StateFS:
    def __init__(self):
        self.IF = IFState()
        self.ID = IDState()
        self.EX = EXState()
        self.MEM = MEMState()
        self.WB = WBState()

# =================================================================================================
# Five-Stage Pipeline Stages
# =================================================================================================

class InstructionFetchStage:
    """Instruction Fetch pipeline stage."""
    def __init__(self, state: StateFS, ins_mem: InsMem):
        self.state = state
        self.ins_mem = ins_mem

    def run(self):
        """Execute instruction fetch."""
        if self.state.IF.nop or self.state.ID.nop or (self.state.ID.hazard_nop and self.state.EX.nop):
            return
        
        instr = self.ins_mem.read_instruction_bin(self.state.IF.PC)[::-1]
        
        if instr == "1" * 32:
            self.state.IF.nop = True
            self.state.ID.nop = True
        else:
            self.state.ID.PC = self.state.IF.PC
            self.state.IF.PC += 4
            self.state.ID.instr = instr

class InstructionDecodeStage:
    """Instruction Decode pipeline stage."""

    def __init__(self, state: StateFS, rf: RegisterFile):
        self.state = state
        self.rf = rf

    def detect_hazard(self, rs):
        if rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem == 0: return 2
        elif rs == self.state.WB.write_reg_addr and self.state.WB.write_enable: return 1
        elif rs == self.state.MEM.write_reg_addr and self.state.MEM.read_mem != 0:
            self.state.ID.hazard_nop = True
            return 1
        return 0

    def read_data(self, rs, forward_signal):
        if forward_signal == 1: return self.state.WB.write_data
        elif forward_signal == 2: return self.state.MEM.alu_result
        else: return self.rf.read_rf_str(rs)

    def run(self):
        """Execute instruction decode."""
        if self.state.ID.nop:
            if not self.state.IF.nop: self.state.ID.nop = False
            return

        self.state.EX.instr = self.state.ID.instr
        self.state.EX.is_I_type = False
        self.state.EX.read_mem = False
        self.state.EX.write_mem = False
        self.state.EX.write_enable = False
        self.state.ID.hazard_nop = False
        self.state.EX.write_reg_addr = "000000"

        opcode = self.state.ID.instr[:7][::-1]
        func3 = self.state.ID.instr[12:15][::-1]

        """Decode R-type instruction."""
        if opcode == OPCODE_STR_R_TYPE:
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]
            
            fwd1 = self.detect_hazard(rs1)
            fwd2 = self.detect_hazard(rs2)
            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, fwd1)
            self.state.EX.read_data_2 = self.read_data(rs2, fwd2)
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.write_enable = True

            func7 = self.state.ID.instr[25:][::-1]
            if func3 == "000": # ADD/SUB
                self.state.EX.alu_op = "00"
                if func7 == "0100000":
                    self.state.EX.read_data_2 = int_to_bin(-bin_to_int(self.state.EX.read_data_2, True))
            elif func3 == "111": self.state.EX.alu_op = "01" # AND
            elif func3 == "110": self.state.EX.alu_op = "10" # OR
            elif func3 == "100": self.state.EX.alu_op = "11" # XOR

            """Decode I-type instruction."""
        elif opcode == OPCODE_STR_I_TYPE_1 or opcode == OPCODE_STR_I_TYPE_2:
            rs1 = self.state.ID.instr[15:20][::-1]
            fwd1 = self.detect_hazard(rs1)
            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.read_data_1 = self.read_data(rs1, fwd1)
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.is_I_type = True
            self.state.EX.imm = self.state.ID.instr[20:][::-1]
            self.state.EX.write_enable = True
            self.state.EX.read_mem = (opcode == OPCODE_STR_I_TYPE_2)

            if func3 == "000": self.state.EX.alu_op = "00"
            elif func3 == "111": self.state.EX.alu_op = "01"
            elif func3 == "110": self.state.EX.alu_op = "10"
            elif func3 == "100": self.state.EX.alu_op = "11"

            """Decode JAL instruction."""
        elif opcode == OPCODE_STR_J_TYPE:
            self.state.EX.imm = ("0" + self.state.ID.instr[21:31] + self.state.ID.instr[20] + self.state.ID.instr[12:20] + self.state.ID.instr[31])[::-1]
            self.state.EX.write_reg_addr = self.state.ID.instr[7:12][::-1]
            self.state.EX.read_data_1 = int_to_bin(self.state.ID.PC)
            self.state.EX.read_data_2 = int_to_bin(4)
            self.state.EX.write_enable = True
            self.state.EX.alu_op = "00"
            self.state.IF.PC = self.state.ID.PC + bin_to_int(self.state.EX.imm, True)
            self.state.ID.nop = True

            """Decode branch instruction."""
        elif opcode == OPCODE_STR_B_TYPE:
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]
            fwd1 = self.detect_hazard(rs1)
            fwd2 = self.detect_hazard(rs2)
            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, fwd1)
            self.state.EX.read_data_2 = self.read_data(rs2, fwd2)
            
            val1 = bin_to_int(self.state.EX.read_data_1, True)
            val2 = bin_to_int(self.state.EX.read_data_2, True)
            diff = val1 - val2

            self.state.EX.imm = ("0" + self.state.ID.instr[8:12] + self.state.ID.instr[25:31] + self.state.ID.instr[7] + self.state.ID.instr[31])[::-1]

            if (diff == 0 and func3 == "000") or (diff != 0 and func3 == "001"):
                self.state.IF.PC = self.state.ID.PC + bin_to_int(self.state.EX.imm, True)
                self.state.ID.nop = True
                self.state.EX.nop = True
            else:
                self.state.EX.nop = True

            """Decode store instruction."""
        elif opcode == OPCODE_STR_S_TYPE:
            rs1 = self.state.ID.instr[15:20][::-1]
            rs2 = self.state.ID.instr[20:25][::-1]
            fwd1 = self.detect_hazard(rs1)
            fwd2 = self.detect_hazard(rs2)
            if self.state.ID.hazard_nop:
                self.state.EX.nop = True
                return

            self.state.EX.rs = rs1
            self.state.EX.rt = rs2
            self.state.EX.read_data_1 = self.read_data(rs1, fwd1)
            self.state.EX.read_data_2 = self.read_data(rs2, fwd2)
            self.state.EX.imm = (self.state.ID.instr[7:12] + self.state.ID.instr[25:])[::-1]
            self.state.EX.is_I_type = True
            self.state.EX.write_mem = True
            self.state.EX.alu_op = "00"

        if self.state.IF.nop:
            self.state.ID.nop = True
        return 1

class ExecutionStage:
    """Execute pipeline stage."""

    def __init__(self, state: StateFS):
        self.state = state

    def run(self):
        """Execute ALU operations."""
        if self.state.EX.nop:
            if not self.state.ID.nop: self.state.EX.nop = False
            return

        op1 = bin_to_int(self.state.EX.read_data_1, True)
        op2_str = self.state.EX.imm if (self.state.EX.is_I_type or self.state.EX.write_mem) else self.state.EX.read_data_2
        op2 = bin_to_int(op2_str, True)
        
        res = 0
        if self.state.EX.alu_op == "00": res = op1 + op2
        elif self.state.EX.alu_op == "01": res = op1 & op2
        elif self.state.EX.alu_op == "10": res = op1 | op2
        elif self.state.EX.alu_op == "11": res = op1 ^ op2

        self.state.MEM.alu_result = int_to_bin(res)
        self.state.MEM.rs = self.state.EX.rs
        self.state.MEM.rt = self.state.EX.rt
        self.state.MEM.read_mem = self.state.EX.read_mem
        self.state.MEM.write_mem = self.state.EX.write_mem
        if self.state.EX.write_mem:
            self.state.MEM.store_data = self.state.EX.read_data_2
        self.state.MEM.write_enable = self.state.EX.write_enable
        self.state.MEM.write_reg_addr = self.state.EX.write_reg_addr

        if self.state.ID.nop: self.state.EX.nop = True

class MemoryAccessStage:
    """Memory Access pipeline stage."""

    def __init__(self, state: StateFS, dmem: DataMem):
        self.state = state
        self.dmem = dmem

    def run(self):
        """Execute memory operations."""
        if self.state.MEM.nop:
            if not self.state.EX.nop: self.state.MEM.nop = False
            return
        
        if self.state.MEM.read_mem:
            self.state.WB.write_data = self.dmem.read_data_bin(self.state.MEM.alu_result)
        elif self.state.MEM.write_mem:
            self.dmem.write_data_bin(self.state.MEM.alu_result, self.state.MEM.store_data)
        else:
            self.state.WB.write_data = self.state.MEM.alu_result
            self.state.MEM.store_data = self.state.MEM.alu_result
        
        self.state.WB.write_enable = self.state.MEM.write_enable
        self.state.WB.write_reg_addr = self.state.MEM.write_reg_addr

        if self.state.EX.nop: self.state.MEM.nop = True

class WriteBackStage:
    """Write Back pipeline stage."""

    def __init__(self, state: StateFS, rf: RegisterFile):
        self.state = state
        self.rf = rf

    def run(self):
        """Execute write back to register file."""
        if self.state.WB.nop:
            if not self.state.MEM.nop: self.state.WB.nop = False
            return
        if self.state.WB.write_enable:
            self.rf.write_rf_str(self.state.WB.write_reg_addr, self.state.WB.write_data)

        if self.state.MEM.nop: self.state.WB.nop = True

class FiveStageCore:
    """Five-stage pipelined RISC-V processor."""

    def __init__(self, io_dir: str, imem: InsMem, dmem: DataMem):
        self.io_dir = io_dir
        # Ensure directory exists
        os.makedirs(io_dir, exist_ok=True)
        self.op_file_path = os.path.join(io_dir, "StateResult_FS.txt")

        self.imem = imem
        self.dmem = dmem
        self.rf = RegisterFile(io_dir, prefix="FS")

        self.state = StateFS()
        self.cycle = 0
        self.inst_count = 0
        self.halted = False

        # Initialize pipeline stages
        self.stages = [
            InstructionFetchStage(self.state, self.imem),
            InstructionDecodeStage(self.state, self.rf),
            ExecutionStage(self.state),
            MemoryAccessStage(self.state, self.dmem),
            WriteBackStage(self.state, self.rf)
        ]

    def step(self):
        """Execute one clock cycle through all pipeline stages."""
        # Halt Check
        if (self.state.IF.nop and self.state.ID.nop and self.state.EX.nop and self.state.MEM.nop and self.state.WB.nop):
            self.halted = True
        
        current_instr = self.state.ID.instr
        
        # Run stages in reverse order (WB to IF) to simulate pipeline register updates
        for stage in reversed(self.stages):
            stage.run()

        self.rf.output_rf_str(self.cycle)
        self.print_state(self.state, self.cycle)

        self.inst_count += int(current_instr != self.state.ID.instr)
        self.cycle += 1

    def print_state(self, state, cycle):
        """Print pipeline state to file."""
        lines = ["-"*70+"\n", f"State after executing cycle: {cycle}\n"]
        
        # Using dataclass helpers to get dictionary views for key-value printing
        for name, obj in [("IF", state.IF), ("ID", state.ID), ("EX", state.EX), ("MEM", state.MEM), ("WB", state.WB)]:
            # We use get_dict_copy if available (for ID, EX, MEM, WB) or standard conversion for IF
            data = obj.get_dict_copy() if hasattr(obj, 'get_dict_copy') else asdict(obj)
            lines.extend([f"{name}.{k}: {v}\n" for k, v in data.items()])
            # lines.append("\n")

        mode = "w" if cycle == 0 else "a"
        with open(self.op_file_path, mode) as f:
            f.writelines(lines)

# =================================================================================================
# METRICS & MAIN
# =================================================================================================

def write_metrics(io_dir: str, ss: SingleStageCore, fs: FiveStageCore):
    ss_metrics = [
        "Single Stage Core Performance Metrics: ",
        f"#Cycles -> {ss.cycle}",
        f"#Instructions -> {ss.inst_count}",
        f"#CPI -> {(ss.cycle) / ss.inst_count if ss.inst_count else 0}",
        f"IPC -> {ss.inst_count / (ss.cycle) if ss.cycle else 0}",
    ]
    
    fs_metrics = [
        "Five Stage Core Performance Metrics:",
        f"#Cycles ->  {fs.cycle}",
        f"#Instructions -> {fs.inst_count}",
        f"#CPI -> {fs.cycle / fs.inst_count if fs.inst_count else 0}",
        f"IPC -> {fs.inst_count / fs.cycle if fs.cycle else 0}",
    ]

    # Individual files
    with open(os.path.join(io_dir, "SingleMetrics.txt"), "w") as f:
        f.write("\n".join(ss_metrics))
    with open(os.path.join(io_dir, "FiveMetrics.txt"), "w") as f:
        f.write("\n".join(fs_metrics))
    # Combined file
    with open(os.path.join(io_dir, "PerformanceMetrics_Result.txt"), "w") as f:
        f.write("\n".join(ss_metrics) + "\n\n" + "\n".join(fs_metrics))

    # Terminal Output
    print("\n" + "\n".join(ss_metrics))
    print("\n" + "\n".join(fs_metrics))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RV32I single and five stage processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()
    io_dir = os.path.abspath(args.iodir)
    print("IO Directory:", io_dir)

    imem = InsMem("Imem", io_dir)

    # =========================================================================
    # SINGLE STAGE PROCESSOR
    # =========================================================================
    dmem_ss = DataMem("SS", io_dir)
    ss_core = SingleStageCore(io_dir, imem, dmem_ss)
    
    # Run until halted
    while not ss_core.halted:
        ss_core.step()

    # Explicitly dump the final halted state
    ss_core.rf.output_rf_int(ss_core.cycle)
    ss_core.print_state(ss_core.next_state, ss_core.cycle)
    ss_core.cycle += 1

    dmem_ss.output_data_mem()

    # =========================================================================
    # FIVE STAGE PROCESSOR
    # =========================================================================
    dmem_fs = DataMem("FS", io_dir)
    fs_core = FiveStageCore(io_dir, imem, dmem_fs)
    
    while not fs_core.halted:
        fs_core.step()
    
    dmem_fs.output_data_mem()
    
    fs_core.inst_count += 1
    
    write_metrics(io_dir, ss_core, fs_core)