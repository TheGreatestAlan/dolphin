import json

from agent_server.integrations.StreamManager import StreamManager
from agent_server.FunctionResponse import FunctionResponse, Status



class InvalidEventException(Exception):
    def __init__(self, invalid_event, valid_events):
        self.invalid_event = invalid_event
        self.valid_events = valid_events
        super().__init__(self._generate_message())

    def _generate_message(self):
        return (f"Invalid event: '{self.invalid_event}'. "
                f"Valid events are: {', '.join(self.valid_events)}")


class LocalDeviceAction:
    valid_events = set()

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self._load_default_events()

    def _load_default_events(self):
        # Preload the valid events
        self.valid_events.add("arrivedHomeEvent")

    def event_alert_action(self, event_name: str, message: str, session_id: str):
        if event_name not in self.valid_events:
            raise InvalidEventException(event_name, self.valid_events)

        # Create the JSON string using the event name and message
        json_data = {
            "eventName": event_name,
            "actionName": "alertAction",
            "actionData": {
                "message": message
            }
        }
        json_string = json.dumps(json_data, indent=2)

        # Prepend the FUNCTION constant from StreamManager
        full_message = f"{StreamManager.FUNCTION}{json_string}"

        # Add to the stream manager
        self.stream_manager.add_to_text_buffer(session_id, full_message)

        try:
            return FunctionResponse(Status.SUCCESS, "event put on stream")
        except Exception as e:
            return FunctionResponse(Status.FAILURE, e)
