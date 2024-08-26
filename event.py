from typing import Dict, Any
import time, json
from dapr.clients import DaprClient
import re
from nanoid import generate
import logging
import traceback


class Event:
    def __init__(self, object_id: str, title: str, text: str, timestamp: time.time, status: bool, data: Dict[str, Any],
                 level: int = 1):
        self.id = generate()
        self.object_id = object_id
        self.title = title
        self.clear_dn = self.object_id + "_" + self.title
        self.text = text
        self.timestamp = timestamp
        self.status = status
        self.data = data
        self.level = level

    def to_dict(self) -> Dict[str, Any]:
        # Replace placeholders in text with corresponding values from data
        formatted_text = self.text
        matches = re.findall(r'\{(\w+)\}', self.text)
        for match in matches:
            if match in self.data:
                formatted_text = formatted_text.replace(f'{{{match}}}', str(self.data[match]))

        return {
            "id": self.id,
            "object_id": self.object_id,
            "dn": self.clear_dn,
            "event_title": self.title,
            "event_text": formatted_text,
            "event_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "status": 1 if self.status else 0,
            "level": self.level,
            "event_extra": json.dumps(self.data),
        }


class EventSender:
    def __init__(self, pubsub_name: str, topic: str):
        """
        Initializes the EventSender and creates a persistent DaprClient instance.

        Args:
            pubsub_name (str): The name of the Dapr pub/sub component.
            topic (str): The name of the topic.
        """
        self.pubsub_name = pubsub_name
        self.topic = topic
        self.client = DaprClient()

    def send_event(self, event: Event):
        """
        Sends an event to a specified topic using Dapr pub/sub.

        Args:
            event (Event): The event to be sent.
        """
        try:
            event_data = json.dumps(event.to_dict())
            self.client.publish_event(pubsub_name=self.pubsub_name, topic_name=self.topic, data=event_data,
                                      data_content_type='application/json')
            logging.debug(f"Event sent to {self.topic}: {event_data}")
        except Exception as e:
            traceback.print_exc()
            logging.error(f"event={event}")
            logging.error(f"Failed to send event to {self.topic}: {e}")

    def close(self):
        """
        Closes the DaprClient instance to free up resources.
        """
        if self.client is not None:
            self.client.close()


eventSender = EventSender("pubsub", "infer_events")
