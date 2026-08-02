"""
Elastic-MTP Side-by-Side Interactive Arena Web Server Launcher.
Starts a local web server rendering the side-by-side prompt arena.
"""
import os
import sys
import webbrowser
import http.server
import socketserver

PORT = 8080
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def launch_server():
    print("=" * 80)
    print("ELASTIC-MTP SIDE-BY-SIDE INTERACTIVE ARENA WEB SERVER")
    print("=" * 80)
    print(f"[OK] Serving Arena at: http://localhost:{PORT}/side_by_side_chat.html")
    print("Opening web browser...")
    
    webbrowser.open(f"http://localhost:{PORT}/side_by_side_chat.html")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    launch_server()
