import psutil
import ctypes
from ctypes import wintypes
import time
import os
import platform
import subprocess
import tempfile
from server import PromptServer

class AnyType(str):
    """用于表示任意类型的特殊类，在类型比较时总是返回相等"""
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class VRAMCleanup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any, {}),
                "offload_model": ("BOOLEAN", {"default": True}),
                "offload_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "Memory Management"

    def empty_cache(self, anything, offload_model, offload_cache, unique_id=None, extra_pnginfo=None):
        try:
            # 发送信号到前端
            PromptServer.instance.send_sync("memory_cleanup", {
                "type": "cleanup_request",
                "data": {
                    "unload_models": offload_model,
                    "free_memory": offload_cache
                }
            })
            print("已发送内存清理信号")
        except Exception as e:
            print(f"发送内存清理信号失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
        time.sleep(1) 
        return (anything,)


class RAMCleanup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any, {}),
                "clean_file_cache": ("BOOLEAN", {"default": True, "label": "清理文件缓存"}),
                "clean_processes": ("BOOLEAN", {"default": True, "label": "清理进程内存"}),
                "clean_dlls": ("BOOLEAN", {"default": True, "label": "清理未使用DLL"}),
                "retry_times": ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "label": "重试次数"
                }),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "clean_ram"
    CATEGORY = "Memory Management"

    def get_ram_usage(self):
        memory = psutil.virtual_memory()
        return memory.percent, memory.available / (1024 * 1024) 

    def clean_ram(self, anything, clean_file_cache, clean_processes, clean_dlls, retry_times, unique_id=None, extra_pnginfo=None):
        try:
            current_usage, available_mb = self.get_ram_usage()
            print(f"开始清理RAM - 当前使用率: {current_usage:.1f}%, 可用: {available_mb:.1f}MB")
            
            system = platform.system()
            for attempt in range(retry_times):
                
                if clean_file_cache:
                    try:
                        if system == "Windows":
                            ctypes.windll.kernel32.SetSystemFileCacheSize(-1, -1, 0)
                        elif system == "Linux":
                            try:
                                subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], 
                                              check=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                                print("使用sudo清理缓存成功")
                            except Exception as sudo_e:
                                print(f"使用sudo清理缓存失败: {str(sudo_e)}")
                                try:
                                    subprocess.run(["sudo", "sysctl", "vm.drop_caches=3"],
                                                  check=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                                    print("使用sysctl清理缓存成功")
                                except Exception as sysctl_e:
                                    print(f"使用sysctl清理缓存失败: {str(sysctl_e)}")
                                    print("请尝试在终端执行: 'sudo sh -c \"echo 3 > /proc/sys/vm/drop_caches\"'")
                    except Exception as e:
                        print(f"清理文件缓存失败: {str(e)}")
                
                if clean_processes:
                    cleaned_processes = 0
                    if system == "Windows":
                        for process in psutil.process_iter(['pid', 'name']):
                            try:
                                handle = ctypes.windll.kernel32.OpenProcess(
                                    wintypes.DWORD(0x001F0FFF),
                                    wintypes.BOOL(False),
                                    wintypes.DWORD(process.info['pid'])
                                )
                                ctypes.windll.psapi.EmptyWorkingSet(handle)
                                ctypes.windll.kernel32.CloseHandle(handle)
                                cleaned_processes += 1
                            except:
                                continue
                    elif system == "Linux":
                        cleaned_processes = 0

                if clean_dlls:
                    try:
                        if system == "Windows":
                            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                        elif system == "Linux":

                            subprocess.run(["sync"], check=True)
                    except Exception as e:
                        print(f"释放内存资源失败: {str(e)}")

                time.sleep(1)
                current_usage, available_mb = self.get_ram_usage()
                print(f"清理后内存使用率: {current_usage:.1f}%, 可用: {available_mb:.1f}MB")

            print(f"清理完成 - 最终内存使用率: {current_usage:.1f}%, 可用: {available_mb:.1f}MB")

        except Exception as e:
            print(f"RAM清理过程出错: {str(e)}")
            
        return (anything,)
    

WEB_DIRECTORY = "web"        
NODE_CLASS_MAPPINGS = {
    "VRAMCleanup": VRAMCleanup,
    "RAMCleanup": RAMCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMCleanup": "🎈VRAM-Cleanup",
    "RAMCleanup": "🎈RAM-Cleanup",
}
