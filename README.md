<h1 align="center"><img src="./public/logosemfundo.png"></h1>

<h1 align="center"> Ded Security Codes</h1>

```bash
Website:  https://www.dedsecurity.com
Author:   Simon Kinjo
Maintenance:  Simon Kinjo
```

>Ded Security Codes is an artificial intelligence tool 
>developed by Ded Security to assist users through code auto-completion.
---

## Installation

Make sure you have installed the dependencies:

  * `python` 3.7.4
  * `git`

Clone the [source] with `git`:
 ```sh
   git clone https://github.com/dedsecurity/codes
   cd codes
   ```
   
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the libraries.

```bash
pip install -r requirements.txt
```

 [source]: https://github.com/dedsecurity/codes

---

## How to use it ?
```bash
python main.py
```

## Screen
```bash
""" create a web server


    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Serv(BaseHTTPRequestHandler):

        def do_GET(self):
            if self.path == '/':
                self.path = 'index.html' # your file
            try:
                file_to_open = open(self.path[1:]).read()
                self.send_response(200)

            except:
                file_to_open = "File not find"
                self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))

    httpd = HTTPServer(('localhost', 8000), Serv)
    httpd.serve_forever()

""" 
```

---

## Contributing
Feel free to submitting pull requests to us.
## License
[MIT](https://opensource.org/licenses/MIT)
