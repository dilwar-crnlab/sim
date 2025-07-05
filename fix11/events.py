#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event System for Dynamic MCF EON Simulation
File: sim/events.py
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional

class EventType(Enum):
    """Types of simulation events"""
    CONNECTION_ARRIVAL = "arrival"
    CONNECTION_TEARDOWN = "teardown"

@dataclass
class SimulationEvent:
    """
    Simulation event for priority queue
    Supports comparison for heapq operations
    """
    event_time: float
    event_type: EventType
    connection_id: str
    connection_data: Optional[Dict] = None
    
    def __lt__(self, other):
        """For priority queue ordering by event time"""
        return self.event_time < other.event_time
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, SimulationEvent):
            return False
        return (self.event_time == other.event_time and 
                self.connection_id == other.connection_id)
    
    def __repr__(self):
        """String representation for debugging"""
        return (f"SimulationEvent(time={self.event_time:.2f}s, "
                f"type={self.event_type.value}, id={self.connection_id})")