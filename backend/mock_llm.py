# mock_llm.py
from datetime import datetime
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """Abstract interface for LLM implementations"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class MockLLM(LLMInterface):
    """Mock LLM implementation for testing purposes"""
    
    def generate_response(self, prompt: str) -> str:
        """Generate mock responses based on prompt content"""
        prompt_lower = prompt.lower()
        
        if "weekly report" in prompt_lower:
            return self._generate_weekly_report()
        elif "project status" in prompt_lower:
            return self._generate_project_status()
        elif "blockers" in prompt_lower:
            return self._generate_blockers_summary()
        elif "mobile app" in prompt_lower:
            return self._answer_mobile_app_question()
        elif "achievements" in prompt_lower:
            return self._answer_achievements_question()
        elif "status" in prompt_lower:
            return self._answer_status_question()
        else:
            return "I can help you with weekly reports, project status, blockers summary, or answer specific questions about the team's work."
    
    def _generate_weekly_report(self) -> str:
        return f"""
## Weekly Team Report - Week of {datetime.now().strftime("%B %d, %Y")}

### Key Achievements
- Authentication system integration completed ahead of schedule
- Mobile app UI/UX improvements showing positive user feedback
- Database optimization reduced query times by 40%
- Successfully onboarded 2 new team members

### Current Focus Areas
- API development for mobile integration
- Payment system implementation
- Performance monitoring dashboard
- User onboarding flow improvements

### Upcoming Priorities
- Complete payment gateway integration
- Finalize mobile app beta testing
- Implement real-time notifications
- Prepare for Q4 feature rollout

### Resource Needs
- Additional QA support for mobile testing
- DevOps consultation for infrastructure scaling

### Team Velocity
- 85% of planned tasks completed
- 2 critical features delivered
- 1 major blocker resolved
        """
    
    def _generate_project_status(self) -> str:
        return """
## Project Status Overview

### On Track Projects (70%)
- **Mobile App Development**: 85% complete, beta testing in progress
- **API Integration**: 90% complete, final testing phase
- **User Dashboard**: 80% complete, UI polish remaining
- **Database Optimization**: 95% complete, performance improvements delivered

### At Risk Projects (20%)
- **Payment System**: 60% complete, integration challenges with PCI compliance
- **Analytics Dashboard**: 50% complete, resource constraints affecting timeline

### Blocked Projects (10%)
- **Third-party Integration**: Waiting for vendor API access approval

### Recent Completions
- User authentication system
- Mobile responsive design
- Load balancer configuration
        """
    
    def _generate_blockers_summary(self) -> str:
        return """
## Current Blockers & Dependencies

### High Priority Blockers
1. **Third-party API Access**: Marketing team waiting for vendor approval (3 days)
2. **Database Migration**: Requires infrastructure team coordination (scheduled next week)
3. **Payment Gateway Integration**: Legal compliance review needed (in progress)

### Medium Priority Blockers
1. **Mobile Testing Resources**: Need additional QA support for comprehensive testing
2. **Performance Optimization**: Requires senior developer review and approval

### Recently Resolved
- GDPR compliance documentation completed
- Load balancer configuration issues fixed
- User authentication bugs resolved

### Action Items
- Follow up with vendor on API access timeline
- Schedule infrastructure team meeting
- Escalate QA resource request to management
        """
    
    def _answer_mobile_app_question(self) -> str:
        return """
**Mobile App Status Update:**

The mobile app is progressing well with 85% completion. Key highlights:
- UI/UX improvements have been implemented with positive user feedback
- Beta testing is currently in progress with 50 signed-up users
- Some UI inconsistencies identified during QA testing are being addressed
- Integration with the new authentication system is complete
- Next phase focuses on payment UI components and final testing

**Timeline:** Expected to complete beta testing by end of next week, with full release targeted for Q4.
        """
    
    def _answer_achievements_question(self) -> str:
        return """
**Key Achievements This Week:**

1. **Technical Deliverables:**
   - Completed new user authentication flow (reduced from 5 to 3 steps)
   - Achieved 40% improvement in database query performance
   - Deployed real-time monitoring dashboard with health alerts

2. **Product Progress:**
   - Finished user research with 15 customers, providing valuable insights
   - Mobile app beta ready with 50 users signed up for testing
   - Email campaign launched with 25% higher open rates

3. **Process Improvements:**
   - Created automated test suite for user dashboard
   - Established Q4 content strategy and launch plans
   - Resolved load balancer issues preventing system downtime

**Impact:** These achievements position the team well for the upcoming Q4 feature rollout.
        """
    
    def _answer_status_question(self) -> str:
        return """
**Overall Team Status:**

**Velocity:** Strong - 85% of planned tasks completed this week
**Morale:** High - team is motivated and collaborative
**Resource Utilization:** Optimal - all team members actively contributing

**By Department:**
- **Engineering:** On track with technical deliverables
- **Product:** Successfully gathering user insights and planning
- **QA:** Proactive testing with good bug detection rate
- **Marketing:** Showing improved campaign performance
- **DevOps:** Infrastructure stable and monitoring improved

**Areas of Focus:** Resource allocation for mobile testing and payment system integration.
        """