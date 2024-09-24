handleClick = event => {
event.preventDefault();

    if (this.props.onClick && typeof this.props.onClick 'function') {
    return this.props.onClick(event);
    }

    return event;
}

handleClick = event => {
    event.preventDefault();