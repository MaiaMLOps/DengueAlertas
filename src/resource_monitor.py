import time
import json
import os
import subprocess
import psutil

LOG_DIR = "/opt/dengue-api/logs"
LOG_FILE = os.path.join(LOG_DIR, "system_resources.jsonl")
INTERVAL_SECONDS = 10

SERVICES = [
    ("mlflow", "mlflow.service"),
    ("api_prod", "dengue-api-prod.service"),
    ("api_staging", "dengue-api-staging.service"),
    ("webhook", "dengue-webhook.service"),
]

def get_service_pids(service_unit):
    """Obtiene el PID principal y todos los procesos hijos de un servicio systemd"""
    try:
        out = subprocess.check_output(
            ["systemctl", "show", "-p", "MainPID", service_unit],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        main_pid = int(out.split("=")[1])
        if main_pid <= 0:
            return []
        
        proc = psutil.Process(main_pid)
        pids = [proc] + proc.children(recursive=True)
        return pids
    except Exception:
        return []

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"[START] Iniciando Monitor Completo (CPU, RAM, Disk I/O, Net I/O) cada {INTERVAL_SECONDS}s...")

    # Primera lectura para calibrar cpu_percent y contadores de I/O
    psutil.cpu_percent(interval=None)
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    prev_time = time.time()

    time.sleep(1)

    while True:
        try:
            start_tick = time.time()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            # 1. Metricas Generales del Sistema (CPU, Load, RAM)
            cpu_sys = psutil.cpu_percent(interval=None)
            mem_sys = psutil.virtual_memory()
            load_1m = os.getloadavg()[0]

            # 2. Metricas de I/O de Disco y Red (Velocidad en KB/s)
            curr_disk = psutil.disk_io_counters()
            curr_net = psutil.net_io_counters()
            now_time = time.time()
            dt = max(0.001, now_time - prev_time)

            disk_read_kb_s = round((curr_disk.read_bytes - prev_disk.read_bytes) / (dt * 1024), 2)
            disk_write_kb_s = round((curr_disk.write_bytes - prev_disk.write_bytes) / (dt * 1024), 2)
            net_in_kb_s = round((curr_net.bytes_recv - prev_net.bytes_recv) / (dt * 1024), 2)
            net_out_kb_s = round((curr_net.bytes_sent - prev_net.bytes_sent) / (dt * 1024), 2)

            prev_disk = curr_disk
            prev_net = curr_net
            prev_time = now_time

            # 3. Metricas Granulares por Servicio
            services_data = {}
            for key, unit in SERVICES:
                procs = get_service_pids(unit)
                if not procs:
                    services_data[key] = {
                        "status": "inactive",
                        "cpu_percent": 0.0,
                        "ram_rss_mb": 0.0,
                        "num_processes": 0
                    }
                    continue

                total_cpu = 0.0
                total_rss_bytes = 0

                for p in procs:
                    try:
                        total_cpu += p.cpu_percent(interval=None)
                        total_rss_bytes += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                services_data[key] = {
                    "status": "active",
                    "cpu_percent": round(total_cpu, 2),
                    "ram_rss_mb": round(total_rss_bytes / (1024 * 1024), 2),
                    "num_processes": len(procs)
                }

            # 4. Ensamblar registro JSON
            record = {
                "timestamp": now_iso,
                "system": {
                    "cpu_percent": round(cpu_sys, 2),
                    "load_avg_1m": round(load_1m, 2),
                    "ram_used_mb": round(mem_sys.used / (1024 * 1024), 2),
                    "ram_total_mb": round(mem_sys.total / (1024 * 1024), 2),
                    "ram_percent": round(mem_sys.percent, 2)
                },
                "io": {
                    "disk_read_kb_s": disk_read_kb_s,
                    "disk_write_kb_s": disk_write_kb_s,
                    "net_in_kb_s": net_in_kb_s,
                    "net_out_kb_s": net_out_kb_s
                },
                "services": services_data
            }

            # 5. Escribir en archivo JSONL
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as e:
            pass

        # Mantener el ritmo constante
        elapsed = time.time() - start_tick
        sleep_time = max(0.5, INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
