from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.contrib.auth import authenticate
import json

class AuthenticationTestCase(TestCase):
    def setUp(self):
        """Set up test data"""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='TestPass123'
        )
        self.login_url = reverse('login')
        self.dashboard_url = reverse('dashboard')

    def test_user_creation(self):
        """Test that user was created correctly"""
        self.assertEqual(self.user.username, 'testuser')
        self.assertEqual(self.user.email, 'test@example.com')
        self.assertTrue(self.user.check_password('TestPass123'))

    def test_regular_login_success(self):
        """Test regular form login with valid credentials"""
        response = self.client.post(self.login_url, {
            'email': 'test@example.com',
            'password': 'TestPass123'
        })
        
        # Should redirect to dashboard
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.dashboard_url)
        
        # User should be authenticated
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_ajax_login_success(self):
        """Test AJAX login with valid credentials"""
        response = self.client.post(self.login_url, {
            'email': 'test@example.com',
            'password': 'TestPass123'
        }, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        
        # Should return JSON response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['success'])
        self.assertEqual(data['username'], 'testuser')
        self.assertIn('redirect_url', data)

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = self.client.post(self.login_url, {
            'email': 'test@example.com',
            'password': 'wrongpassword'
        })
        
        # Should redirect back to login
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.login_url)

    def test_ajax_login_invalid_credentials(self):
        """Test AJAX login with invalid credentials"""
        response = self.client.post(self.login_url, {
            'email': 'test@example.com',
            'password': 'wrongpassword'
        }, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        
        # Should return JSON error response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    def test_login_nonexistent_user(self):
        """Test login with non-existent user"""
        response = self.client.post(self.login_url, {
            'email': 'nonexistent@example.com',
            'password': 'TestPass123'
        })
        
        # Should redirect back to login
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.login_url)

    def test_ajax_login_nonexistent_user(self):
        """Test AJAX login with non-existent user"""
        response = self.client.post(self.login_url, {
            'email': 'nonexistent@example.com',
            'password': 'TestPass123'
        }, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        
        # Should return JSON error response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    def test_dashboard_access_authenticated(self):
        """Test that authenticated users can access dashboard"""
        # Login first
        self.client.login(username='testuser', password='TestPass123')
        
        # Access dashboard
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['user'].is_authenticated)

    def test_dashboard_access_unauthenticated(self):
        """Test that unauthenticated users are redirected to login"""
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, 302)
        # Django adds a 'next' parameter to the redirect URL, so we check if it starts with the login URL
        self.assertTrue(response.url.startswith(self.login_url))

    def test_logout(self):
        """Test logout functionality"""
        # Login first
        self.client.login(username='testuser', password='TestPass123')
        
        # Logout
        logout_url = reverse('logout')
        response = self.client.get(logout_url)
        
        # Should redirect to login
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('login'))
        
        # User should not be authenticated
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_session_creation_on_login(self):
        """Test that session is created properly on login"""
        response = self.client.post(self.login_url, {
            'email': 'test@example.com',
            'password': 'TestPass123'
        })
        
        # Check that session has last_activity
        self.assertIn('last_activity', response.wsgi_request.session)

    def test_ajax_login_session_creation(self):
        """Test that session is created properly on AJAX login"""
        response = self.client.post(self.login_url, {
            'email': 'test@example.com',
            'password': 'TestPass123'
        }, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        
        # Check that session has last_activity
        self.assertIn('last_activity', response.wsgi_request.session)
