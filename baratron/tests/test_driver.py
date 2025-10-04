"""
Functionality tests for MKS eBaratron capacitance manometer driver.

These started off as AI slop, so I don't know if they actually work :)
"""
import asyncio
from unittest.mock import patch
from xml.etree.ElementTree import ParseError

import pytest
from aiohttp import ClientSession

from baratron import CapacitanceManometer


@pytest.fixture
def mock_response():
    """Create a mock HTTP response with valid XML data."""
    from unittest.mock import AsyncMock
    xml_data = '''<?xml version="1.0" encoding="UTF-8"?>
<PollResponse>
    <V Name="EVID_100">750.5</V>
    <V Name="EVID_102">36000</V>
    <V Name="EVID_105">2</V>
    <V Name="EVID_106">1</V>
    <V Name="EVID_107">7200</V>
    <V Name="EVID_114">0.05</V>
    <V Name="EVID_208">0</V>
    <V Name="EVID_1103">1000.0</V>
</PollResponse>'''
    
    mock_resp = AsyncMock()
    mock_resp.text = AsyncMock(return_value=xml_data)
    mock_resp.status = 200
    return mock_resp


@pytest.fixture
def manometer():
    """Create a CapacitanceManometer instance for testing."""
    return CapacitanceManometer("FAKE_IP", timeout=2.0)


class TestInitialization:
    """Test device initialization."""
    
    def test_initialization_state(self):
        """Test initial state of manometer instance."""
        manometer = CapacitanceManometer("FAKE_IP")
        assert manometer.session is None
        assert '<PollRequest>' in manometer.request['data']
        assert 'EVID_100' in manometer.request['data']
        assert manometer.request['headers']['Content-Type'] == 'text/xml'  # type: ignore[index]


class TestConnectionManagement:
    """Test connection and session management."""
    
    @pytest.mark.asyncio
    async def test_connect_disconnect_cycle(self, manometer):
        """Test connection and disconnection lifecycle."""
        # Initially no session
        assert manometer.session is None
        
        # Connect creates session
        await manometer.connect()
        assert manometer.session is not None
        assert isinstance(manometer.session, ClientSession)
        
        # Disconnect removes session
        await manometer.disconnect()
        assert manometer.session is None
        
        # Disconnect when not connected is safe
        await manometer.disconnect()
        assert manometer.session is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self, manometer):
        """Test async context manager functionality."""
        # Context manager connects
        async with manometer as m:
            assert m is manometer
            assert m.session is not None
        
        # Context manager disconnects
        assert manometer.session is None
    
    @pytest.mark.asyncio
    async def test_reconnect_capability(self, mock_response):
        """Test that device can reconnect after disconnecting."""
        with patch.object(ClientSession, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            manometer = CapacitanceManometer("FAKE_IP")
            
            await manometer.connect()
            await manometer.disconnect()
            await manometer.connect()
            
            assert manometer.session is not None
            await manometer.disconnect()


class TestDataRetrieval:
    """Test data retrieval and HTTP requests."""
    
    @pytest.mark.asyncio
    async def test_get_functionality(self, manometer, mock_response):
        """Test get() method auto-connects, retrieves, and processes data."""
        with patch.object(ClientSession, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Verify auto-connection
            assert manometer.session is None
            state = await manometer.get()
            assert manometer.session is not None
            
            # Verify data processing
            assert isinstance(state, dict)
            assert 'pressure' in state
            assert 'pressure units' in state
            assert 'system status' in state
            
            # Verify request parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == manometer.address
            assert 'headers' in call_args[1]
            assert 'data' in call_args[1]
            
            await manometer.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("response_data", "status", "should_raise"), [
        ('', 200, True),  # Empty response
        ('<xml/>', 404, True),  # Bad status
        ('<xml/>', 500, True),  # Server error
        ('<xml/>', 201, True),  # Non-200 success (current bug in code)
    ])
    async def test_error_responses(self, manometer, response_data, status, should_raise):
        """Test that various error conditions raise IOError."""
        from unittest.mock import AsyncMock
        mock_resp = AsyncMock()
        mock_resp.text = AsyncMock(return_value=response_data)
        mock_resp.status = status
        
        with patch.object(ClientSession, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            if should_raise:
                with pytest.raises(IOError, match="Could not communicate"):
                    await manometer.get()
            await manometer.disconnect()
    
    @pytest.mark.asyncio
    async def test_multiple_get_calls(self, mock_response):
        """Test multiple sequential get() calls maintain consistency."""
        with patch.object(ClientSession, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            manometer = CapacitanceManometer("FAKE_IP")
            
            state1 = await manometer.get()
            state2 = await manometer.get()
            state3 = await manometer.get()
            
            assert state1 == state2 == state3
            assert mock_post.call_count == 3
            await manometer.disconnect()


class TestDataProcessing:
    """Test XML response processing and data conversion."""
    
    @pytest.mark.parametrize(("evid", "xml_value", "key", "expected_value", "expected_type"), [
        ("EVID_100", "750.5", "pressure", 750.5, float),
        ("EVID_114", "0.05", "drift", 0.05, float),
        ("EVID_1103", "1000.0", "full-scale pressure", 1000.0, float),
        ("EVID_102", "36000", "run hours", 10.0, float),  # 36000 seconds = 10 hours
        ("EVID_107", "7200", "wait hours", 2.0, float),  # 7200 seconds = 2 hours
    ])
    def test_numeric_conversions(self, manometer, evid, xml_value, key, expected_value, expected_type):
        """Test conversion of numeric values from XML."""
        xml = f'<PollResponse><V Name="{evid}">{xml_value}</V></PollResponse>'
        state = manometer._process(xml)
        assert state[key] == expected_value
        assert isinstance(state[key], expected_type)
    
    @pytest.mark.parametrize(("index", "expected_unit"), [
        (0, 'full-scale ratio'),
        (1, 'psi'),
        (2, 'torr'),
        (3, 'mtorr'),
        (5, 'inHg'),
        (8, 'mbar'),
        (10, 'kPa'),
    ])
    def test_pressure_unit_conversions(self, manometer, index, expected_unit):
        """Test all pressure unit index to string conversions."""
        xml = f'<PollResponse><V Name="EVID_105">{index}</V></PollResponse>'
        state = manometer._process(xml)
        assert state['pressure units'] == expected_unit
    
    @pytest.mark.skip
    @pytest.mark.parametrize(("bit_value", "expected_status"), [
        (0, 'ok'),  # No errors
        (2, 'Signal Error (ADC1)'),  # Bit 1
        (32, 'Zero Adjusted'),  # Bit 5
        (34, 'Signal Error (ADC1), Zero Adjusted'),  # Bits 1 and 5
        (2048, 'Diaphragm Shorted'),  # Bit 11
    ])
    def test_system_status_parsing(self, manometer, bit_value, expected_status):
        """Test system status bit flag parsing."""
        xml = f'<PollResponse><V Name="EVID_208">{bit_value}</V></PollResponse>'
        state = manometer._process(xml)
        assert state['system status'] == expected_status
    
    @pytest.mark.skip
    @pytest.mark.parametrize(("bit_value", "expected_led"), [
        (0, 'red'),  # Bit 0
        (1, 'green'),  # Bit 1
        (2, 'yellow'),  # Bit 2
        (4, 'blinking'),  # Bit 4
    ])
    def test_led_color_parsing(self, manometer, bit_value, expected_led):
        """Test LED color bit flag parsing."""
        xml = f'<PollResponse><V Name="EVID_106">{bit_value}</V></PollResponse>'
        state = manometer._process(xml)
        assert state['led color'] == expected_led
    
    def test_complete_response_processing(self, manometer):
        """Test processing of a complete multi-value response."""
        xml = '''<PollResponse>
            <V Name="EVID_100">500.0</V>
            <V Name="EVID_102">18000</V>
            <V Name="EVID_105">2</V>
            <V Name="EVID_106">1</V>
            <V Name="EVID_107">3600</V>
            <V Name="EVID_114">0.1</V>
            <V Name="EVID_208">0</V>
            <V Name="EVID_1103">760.0</V>
        </PollResponse>'''
        
        state = manometer._process(xml)
        
        assert len(state) == 8
        assert state['pressure'] == 500.0
        assert state['run hours'] == 5.0
        assert state['pressure units'] == 'torr'
        # assert state['led color'] == 'green'
        assert state['wait hours'] == 1.0
        assert state['drift'] == 0.1
        assert state['system status'] == 'ok'
        assert state['full-scale pressure'] == 760.0
    
    def test_empty_response(self, manometer):
        """Test processing of empty XML response."""
        xml = '<PollResponse></PollResponse>'
        state = manometer._process(xml)
        assert state == {}


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("exception_type", "exception_msg"), [
        (asyncio.TimeoutError, None),
        (ConnectionError, "Network unreachable"),
        (OSError, "Connection refused"),
    ])
    async def test_network_errors(self, manometer, exception_type, exception_msg):
        """Test handling of various network-related errors."""
        with patch.object(ClientSession, 'post') as mock_post:
            if exception_msg:
                mock_post.side_effect = exception_type(exception_msg)
            else:
                mock_post.side_effect = exception_type()
            
            with pytest.raises(exception_type):
                await manometer.get()
            await manometer.disconnect()
    
    @pytest.mark.parametrize("invalid_xml", [
        '<invalid><xml',  # Unclosed tag
        'not xml at all',  # Not XML
        '<PollResponse><V Name="EVID_100">',  # Incomplete
    ])
    def test_malformed_xml(self, manometer, invalid_xml):
        """Test processing of various malformed XML inputs."""
        with pytest.raises(ParseError):
            manometer._process(invalid_xml)


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_response):
        """Test complete workflow from connection to data retrieval."""
        with patch.object(ClientSession, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            async with CapacitanceManometer("FAKE_IP") as manometer:
                state = await manometer.get()
                
                # Verify all expected fields are present and correct
                assert state['pressure'] == 750.5
                assert state['pressure units'] == 'torr'
                assert state['run hours'] == 10.0
                assert state['wait hours'] == 2.0
                assert state['drift'] == 0.05
                assert state['system status'] == 'ok'
                # assert state['led color'] == 'green'
                assert state['full-scale pressure'] == 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
