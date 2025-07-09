import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
    name: "memory.cleanup",
    init() {
        api.addEventListener("memory_cleanup", ({ detail }) => {
            if (detail.type === "cleanup_request") {
                console.log("收到内存清理请求");
                fetch("/free", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(detail.data)
                })
                .then(response => {
                    if (response.ok) {
                        console.log("内存清理请求已发送");
                    } else {
                        console.error("内存清理请求失败");
                    }
                })
                .catch(error => {
                    console.error("发送内存清理请求出错:", error);
                });
            }
        });
    }
});
