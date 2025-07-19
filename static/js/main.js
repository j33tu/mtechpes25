function togglePasswordVisibility(passwordId, toggleButtonId) {
    const passwordInput = document.getElementById(passwordId);
    const toggleButton = document.getElementById(toggleButtonId);

    if (passwordInput && toggleButton) {
        toggleButton.addEventListener('click', function () {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);


            const icon = this.querySelector('i');
            icon.classList.toggle('bi-eye');
            icon.classList.toggle('bi-eye-slash');
        });
    }
}

function showValidationError(inputId, message) {
    const input = document.getElementById(inputId);
    if (input) {
        input.classList.add('is-invalid');
        let feedback = input.parentElement.querySelector('.invalid-feedback');
        if (feedback) {
            feedback.textContent = message;
        }
    }
}

function clearValidationError(inputId) {
    const input = document.getElementById(inputId);
    if (input) {
        input.classList.remove('is-invalid');
        let feedback = input.parentElement.querySelector('.invalid-feedback');
        if (feedback) {
            feedback.textContent = '';
        }
    }
}

function validateLoginForm() {
    let valid = true;
    const username = document.getElementById('username');
    const password = document.getElementById('password');
    clearValidationError('username');
    clearValidationError('password');

    if (!username.value.trim()) {
        showValidationError('username', 'Username is required.');
        valid = false;
    }
    if (!password.value.trim()) {
        showValidationError('password', 'Password is required.');
        valid = false;
    }
    return valid;
}

function validateRegisterForm() {
    let valid = true;
    const username = document.getElementById('username');
    const email = document.getElementById('email');
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirm_password');
    clearValidationError('username');
    clearValidationError('email');
    clearValidationError('password');
    clearValidationError('confirm_password');

    if (!username.value.trim()) {
        showValidationError('username', 'Username is required.');
        valid = false;
    }
    if (!email.value.trim() || !email.value.includes('@')) {
        showValidationError('email', 'Please enter a valid email address.');
        valid = false;
    }
    if (!password.value.trim()) {
        showValidationError('password', 'Password is required.');
        valid = false;
    }
    if (!confirmPassword.value.trim()) {
        showValidationError('confirm_password', 'Confirm Password is required.');
        valid = false;
    } else if (confirmPassword.value !== password.value) {
        showValidationError('confirm_password', 'Passwords do not match.');
        valid = false;
    }
    return valid;
}

function initFormValidation(formId) {
    const form = document.getElementById(formId);
    if (form) {
        form.addEventListener('submit', function (event) {
            let valid = true;
            if (formId === 'loginForm') {
                valid = validateLoginForm();
            } else if (formId === 'registerForm') {
                valid = validateRegisterForm();
            } else {
                valid = form.checkValidity();
            }
            if (!valid) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    }
}


document.addEventListener('DOMContentLoaded', function () {

    console.log('Application initialized');
    initFormValidation('loginForm');
    initFormValidation('registerForm');
});