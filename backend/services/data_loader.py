# data_loader.py
from typing import List

from core import Update


class DataLoader:
    """Handles loading and managing mock data"""

    @staticmethod
    def get_mock_updates() -> List[Update]:
        """Returns a list of mock employee updates"""
        mock_data = [
            {
                "employee": "Sarah Chen",
                "role": "Frontend Developer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-08",
                "update": "Completed the new user authentication flow. The login/signup process is now much smoother and we've reduced the steps from 5 to 3. Also fixed the mobile responsive issues on the dashboard. Currently working on integrating the new design system components. Planning to start the payment UI components next week.",
            },
            {
                "employee": "Mike Rodriguez",
                "role": "Backend Developer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-08",
                "update": "Finished the API endpoints for user management. All CRUD operations are working and documented. The database queries are now 40% faster after optimization. Started working on the payment gateway integration - it's more complex than expected due to PCI compliance requirements. Might need legal review.",
            },
            {
                "employee": "Lisa Wang",
                "role": "Product Manager",
                "department": "Product",
                "manager": "Jennifer Adams",
                "date": "2025-07-08",
                "update": "Completed user research interviews with 15 customers. Key insights: users want faster onboarding and better mobile experience. Updated the roadmap based on findings. The mobile app beta is ready for testing - we have 50 users signed up. Also coordinated with marketing on the Q4 launch strategy.",
            },
            {
                "employee": "David Kim",
                "role": "DevOps Engineer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-08",
                "update": "Deployed the new monitoring dashboard. We now have real-time alerts for system health. Fixed the load balancer issue that was causing intermittent downtime. Working on the database migration plan for next month. The new infrastructure should handle 10x our current load.",
            },
            {
                "employee": "Emma Thompson",
                "role": "QA Engineer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-08",
                "update": "Completed testing for the authentication system - found 3 minor bugs that are now fixed. Created automated test suite for the user dashboard. The mobile app testing is in progress, found some UI inconsistencies that need attention. Will need more resources for the upcoming payment system testing.",
            },
            {
                "employee": "Alex Johnson",
                "role": "Marketing Manager",
                "department": "Marketing",
                "manager": "Robert Smith",
                "date": "2025-07-08",
                "update": "Launched the new email campaign, seeing 25% higher open rates. The content strategy for Q4 is finalized. Working with the product team on beta user feedback collection. Still waiting for the third-party integration approval from the vendor - this is blocking our analytics dashboard integration.",
            },
            {
                "employee": "Tom Wilson",
                "role": "Backend Developer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-09",
                "update": "Implemented the new caching layer for the API. Response times improved by 30%. Working on the webhook system for real-time notifications. The payment gateway integration is proving challenging - we need to handle multiple payment providers. Also helping with the database migration testing.",
            },
            {
                "employee": "Jessica Brown",
                "role": "UX Designer",
                "department": "Design",
                "manager": "Maria Garcia",
                "date": "2025-07-09",
                "update": "Completed the user flow redesign for the onboarding process. User testing showed 40% improvement in completion rates. Working on the payment flow designs - need to ensure PCI compliance in the UI. Created high-fidelity mockups for the mobile app's new features.",
            },
            {
                "employee": "Chris Lee",
                "role": "Senior Developer",
                "department": "Engineering",
                "manager": "Jennifer Adams",
                "date": "2025-07-09",
                "update": "Code review completed for the authentication system. Performance optimization work is ongoing - identified several bottlenecks in the data processing pipeline. Leading the technical planning for the Q4 architecture changes. Mentoring the two new developers on our coding standards.",
            },
        ]

        return [Update.from_dict(data) for data in mock_data]

    @staticmethod
    def get_additional_mock_updates() -> List[Update]:
        """Returns additional mock updates for testing different scenarios"""
        additional_data = [
            {
                "employee": "Sarah Chen",
                "role": "Frontend Developer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-10",
                "update": "Started working on the payment UI components. The design system integration is complete and working well. Fixed a critical bug in the mobile navigation. The new components are 50% complete and should be ready for testing by Friday.",
            },
            {
                "employee": "Mike Rodriguez",
                "role": "Backend Developer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-10",
                "update": "Payment gateway integration is now 80% complete. Resolved the PCI compliance issues with help from the legal team. The webhook system is functional and ready for testing. Performance testing shows all endpoints are within acceptable response times.",
            },
            {
                "employee": "Lisa Wang",
                "role": "Product Manager",
                "department": "Product",
                "manager": "Jennifer Adams",
                "date": "2025-07-10",
                "update": "Beta user feedback is very positive - 90% satisfaction rate. Identified 3 new feature requests that we should consider for the next sprint. The Q4 launch timeline is confirmed. Coordinated with support team for the upcoming user onboarding.",
            },
        ]

        return [Update.from_dict(data) for data in additional_data]

    @staticmethod
    def get_mock_updates_with_blockers() -> List[Update]:
        """Returns mock updates that include various blockers and challenges"""
        blocker_data = [
            {
                "employee": "Alex Johnson",
                "role": "Marketing Manager",
                "department": "Marketing",
                "manager": "Robert Smith",
                "date": "2025-07-10",
                "update": "Still blocked on the third-party integration - vendor hasn't responded to our API access request. This is now affecting our analytics dashboard timeline. Meanwhile, prepared the marketing materials for the Q4 launch. The email campaign performance continues to be strong.",
            },
            {
                "employee": "Emma Thompson",
                "role": "QA Engineer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-10",
                "update": "Testing is behind schedule due to resource constraints. The mobile app testing revealed 5 critical bugs that need immediate attention. Automated test suite is 70% complete. Really need additional QA support to meet the Q4 deadline.",
            },
            {
                "employee": "David Kim",
                "role": "DevOps Engineer",
                "department": "Engineering",
                "manager": "Chris Lee",
                "date": "2025-07-10",
                "update": "Database migration is ready but waiting for the maintenance window approval. Infrastructure scaling is on hold pending budget approval. The monitoring dashboard is working well - caught 2 potential issues before they became problems.",
            },
        ]

        return [Update.from_dict(data) for data in blocker_data]
