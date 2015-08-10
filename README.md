# baratron

Python driver and command line tool for ToolWeb-enabled [MKS eBaratron capacitance manometers](http://www.mksinst.com/product/category.aspx?CategoryID=72).

<p align="center">
  <img src="http://www.mksinst.com/images/pdimages/627c.jpg" />
</p>

Installation
============

```
pip install baratron
```

If you don't like pip, you can also install from source:

```
git clone https://github.com/numat/baratron.git
cd baratron
python setup.py install
```

Usage
=====

###Command Line

To test your connection and stream real-time data, use the command-line
interface. You can read the state with

```
$ baratron 192.168.1.100
{
  "connected": true,
  "full-scale pressure": 1000.0,
  "ip": "192.168.1.100",
  "led color": "green",
  "pressure": 746.07,
  "pressure units": "torr",
  "run hours": 29.66,
  "system status": "ok",
  "wait hours": 0.0
}
```

or stream a table of data with the `--stream` flag. See `baratron --help`
for more.

###Python (Asynchronous)

Asynchronous programming allows us to send out all of our requests in
parallel, and then handle responses as they trickle in. For more information,
read through [krondo's twisted introduction](http://krondo.com/?page_id=1327).

```python
from baratron import CapacitanceManometer
from tornado.ioloop import IOLoop, PeriodicCallback

def on_response(response):
    """This function gets run whenever a device responds."""
    print(response)

def loop():
    """This function will be called in an infinite loop by tornado."""
    for sensor in sensors:
        sensor.get(on_response)

# As an example, this is 100 sensors between 192.168.1.100 and 192.168.1.199
sensors = [CapacitanceManometer('192.168.1.{}'.format(i)) for i in range(100, 200)]

PeriodicCallback(loop, 500).start()
IOLoop.current().start()
```

This looks more complex, but the advantages are well worth it at scale.
Essentially, sleeping is replaced by scheduling functions with tornado. This
allows your code to do other things while waiting for responses.
